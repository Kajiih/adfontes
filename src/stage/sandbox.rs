//! Confined filesystem staging policy, symlink/reparse guards, and atomic `0600` file writes.

use std::ffi::{OsStr, OsString};
use std::fs;
#[cfg(unix)]
use std::fs::File;
#[cfg(not(unix))]
use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Read, Write as _};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::MetadataExt as WindowsMetadataExt;
#[cfg(any(windows, test))]
use std::path::Prefix;
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use rustix::fs::{CWD, Mode, OFlags, fchmod, mkdirat, openat};
#[cfg(unix)]
use rustix::io::Errno;
use tempfile::Builder as TempFileBuilder;

use crate::model::AdfontesError;

const MAX_LOCAL_PDF_BYTES: u64 = 100 * 1024 * 1024;
const DEFAULT_STAGING_NAME: &str = "adfontes";
#[cfg(unix)]
const LABEL_STAGING_ROOT: &str = "staging root";
const LABEL_STAGING_PATH: &str = "staging path";
#[cfg(unix)]
const FILE_MODE_OWNER_ONLY: u32 = 0o600;
#[cfg(unix)]
const MODE_GROUP_OTHER_MASK: u32 = 0o077;
#[cfg(unix)]
const DIR_OPEN_FLAGS: OFlags = OFlags::from_bits_retain(
    OFlags::RDONLY.bits()
        | OFlags::DIRECTORY.bits()
        | OFlags::NOFOLLOW.bits()
        | OFlags::CLOEXEC.bits(),
);
#[cfg(windows)]
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;

const FORBIDDEN_PATH_SEGMENTS: &[&str] = &[".git", ".jj", ".hg", ".ssh", ".gnupg", ".aws", ".kube"];
const FORBIDDEN_HOME_ROOTS: &[&[&str]] = &[
    &[".gemini", "config"],
    &[".agents"],
    &[".config"],
    &[".local"],
];

/// Boundary-parsed filesystem confinement and PDF slicing policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingPolicy {
    /// Process temp directory (`std::env::temp_dir()`).
    pub temp_dir: PathBuf,
    /// Conventional Unix `/tmp` root (resolves to `/private/tmp` on macOS).
    pub system_tmp_dir: PathBuf,
    /// User home directory (`HOME` / `USERPROFILE`), when available.
    pub home_dir: Option<PathBuf>,
    /// Optional operator-configured default staging root (`ADFONTES_STAGING_DIR`).
    pub configured_staging_dir: Option<PathBuf>,
    /// Additional operator-configured allowed staging roots (`ADFONTES_EXTRA_ALLOWED_ROOTS`).
    pub extra_allowed_roots: Vec<PathBuf>,
    /// Additional operator-configured absolute forbidden path prefixes
    /// (`ADFONTES_FORBIDDEN_PATH_PREFIXES`).
    pub forbidden_path_prefixes: Vec<PathBuf>,
    /// Disables local PDF slicing when `ADFONTES_DISABLE_PDF_SLICE` is truthy.
    pub disable_pdf_slice: bool,
    /// Effective Unix UID verified against existing staging directories.
    #[cfg(unix)]
    pub effective_uid: u32,
}

impl StagingPolicy {
    /// Constructs a [`StagingPolicy`] by reading environment variables once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        let home_dir = std::env::var_os("HOME")
            .filter(|val| !val.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|val| !val.is_empty()))
            .map(PathBuf::from);
        let configured_staging_dir = std::env::var_os("ADFONTES_STAGING_DIR")
            .filter(|val| !val.is_empty())
            .map(PathBuf::from);
        let extra_allowed_roots =
            std::env::var_os("ADFONTES_EXTRA_ALLOWED_ROOTS").map_or_default(|raw| {
                std::env::split_paths(&raw)
                    .filter(|path| !path.as_os_str().is_empty())
                    .collect()
            });
        let forbidden_path_prefixes = std::env::var_os("ADFONTES_FORBIDDEN_PATH_PREFIXES")
            .map_or_default(|raw| {
                std::env::split_paths(&raw)
                    .filter(|path| !path.as_os_str().is_empty())
                    .collect()
            });
        let disable_pdf_slice =
            std::env::var("ADFONTES_DISABLE_PDF_SLICE").is_ok_and(|val| is_truthy_env(&val));

        Self {
            temp_dir: std::env::temp_dir(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir,
            configured_staging_dir,
            extra_allowed_roots,
            forbidden_path_prefixes,
            disable_pdf_slice,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }
}

/// Validates that `requested` (or the default staging directory) satisfies filesystem confinement
/// policy without creating any directories or files.
///
/// Callers should invoke this before performing outbound network retrieval so forbidden paths or
/// misconfigured `ADFONTES_STAGING_DIR` values fail fast without leaking egress.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the target path escapes allowed roots, targets
/// configured forbidden prefixes, VCS/credential directories, or forbidden agent config roots,
/// traverses an existing symlink or non-directory, or contains control characters or `..`
/// traversal segments.
pub fn validate_staging_dir_policy(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    let plan = plan_confined_staging_dir(requested, policy)?;
    verify_existing_confined_path(&plan.root, &plan.relative_tail, &plan.expected_path, policy)?;
    let verified_prefix = canonicalize_strict_existing_prefix(&plan.expected_path, policy)?;
    reject_forbidden_path(&verified_prefix, policy)?;
    ensure_within_root(&verified_prefix, &plan.root.canonical, LABEL_STAGING_PATH)
}

/// Validates and creates (if needed) a confined staging directory with `0700` Unix permissions and
/// UID ownership enforcement on every directory below the allowed root.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the requested path escapes allowed roots, targets
/// configured forbidden prefixes, VCS/credential directories, or forbidden agent config roots,
/// traverses a symlink, contains control characters or `..` traversal segments, or encounters a
/// directory owned by another Unix user.
pub fn resolve_confined_staging_dir(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    let plan = plan_confined_staging_dir(requested, policy)?;
    create_and_verify_confined_dir(&plan.root, &plan.relative_tail, &plan.expected_path, policy)
}

pub(super) struct ConfinedPdfBytes {
    pub(super) canonical_path: PathBuf,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn read_bounded_confined_pdf(
    pdf_path: &Path,
    policy: &StagingPolicy,
) -> Result<ConfinedPdfBytes, AdfontesError> {
    let abs_pdf = normalize_candidate_path(pdf_path)?;
    reject_forbidden_path(&abs_pdf, policy)?;

    let allowed_roots = collect_allowed_roots(policy)?;
    let matched = match_allowed_root(&abs_pdf, &allowed_roots, policy)?;
    let Some((file_name, parent_dirs)) = matched.tail.split_last() else {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF path `{}` must point to a file inside a staging directory",
                abs_pdf.display()
            ),
        });
    };

    let canonical_pdf = join_normal_segments(&matched.root.canonical, &matched.tail);
    reject_forbidden_path(&canonical_pdf, policy)?;

    #[cfg(unix)]
    let file =
        open_confined_pdf_unix(matched.root, parent_dirs, file_name, &canonical_pdf, policy)?;

    #[cfg(not(unix))]
    let (file, canonical_pdf) = {
        let _ = file_name;
        verify_non_unix_no_symlink_ancestors(&matched.root.canonical, parent_dirs, &canonical_pdf)?;
        let lstat = fs::symlink_metadata(&canonical_pdf).map_err(|source| AdfontesError::Io {
            context: format!("statting PDF `{}`", canonical_pdf.display()),
            source,
        })?;
        if is_symlink_or_reparse(&lstat) || !lstat.is_file() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF path `{}` must be a regular non-symlink file",
                    canonical_pdf.display()
                ),
            });
        }
        let verified_pdf = finalize_canonical_confined_path(
            &canonical_pdf,
            &matched.root.canonical,
            policy,
            "canonicalizing PDF",
            "canonical PDF path",
        )?;
        let opened = OpenOptions::new()
            .read(true)
            .open(&verified_pdf)
            .map_err(|source| AdfontesError::Io {
                context: format!("opening PDF `{}`", verified_pdf.display()),
                source,
            })?;
        (opened, verified_pdf)
    };

    let fstat = file.metadata().map_err(|source| AdfontesError::Io {
        context: format!("inspecting opened PDF `{}`", canonical_pdf.display()),
        source,
    })?;
    if !fstat.is_file() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF path `{}` must be a regular file (rejected FIFO, device, or directory)",
                canonical_pdf.display()
            ),
        });
    }
    if fstat.len() > MAX_LOCAL_PDF_BYTES {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF file `{}` ({} bytes) exceeds {MAX_LOCAL_PDF_BYTES}-byte safety ceiling",
                canonical_pdf.display(),
                fstat.len()
            ),
        });
    }

    let mut bytes = Vec::new();
    file.take(MAX_LOCAL_PDF_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| AdfontesError::Io {
            context: format!("reading PDF `{}`", canonical_pdf.display()),
            source,
        })?;
    if bytes.len() as u64 > MAX_LOCAL_PDF_BYTES {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF `{}` grew beyond {MAX_LOCAL_PDF_BYTES}-byte safety ceiling while reading",
                canonical_pdf.display()
            ),
        });
    }
    Ok(ConfinedPdfBytes {
        canonical_path: canonical_pdf,
        bytes,
    })
}

pub(super) fn write_file_atomic(destination: &Path, bytes: &[u8]) -> Result<(), AdfontesError> {
    let parent = destination
        .parent()
        .ok_or_else(|| AdfontesError::SafetyBlock {
            reason: format!(
                "destination `{}` has no parent directory",
                destination.display()
            ),
        })?;
    let mut temp = TempFileBuilder::new()
        .prefix(".adfontes-tmp-")
        .tempfile_in(parent)
        .map_err(|source| AdfontesError::Io {
            context: format!("creating temp file in `{}`", parent.display()),
            source,
        })?;
    #[cfg(unix)]
    {
        let perms = fs::Permissions::from_mode(FILE_MODE_OWNER_ONLY);
        temp.as_file()
            .set_permissions(perms)
            .map_err(|source| AdfontesError::Io {
                context: format!("setting 0600 permissions in `{}`", parent.display()),
                source,
            })?;
    }
    temp.write_all(bytes).map_err(|source| AdfontesError::Io {
        context: format!("writing temp file in `{}`", parent.display()),
        source,
    })?;
    temp.as_file()
        .sync_data()
        .map_err(|source| AdfontesError::Io {
            context: format!("syncing temp file in `{}`", parent.display()),
            source,
        })?;
    temp.persist(destination)
        .map_err(|persist_error| AdfontesError::Io {
            context: format!("persisting `{}`", destination.display()),
            source: persist_error.error,
        })?;
    Ok(())
}

pub(super) fn remove_file_if_present(path: &Path) -> Result<(), AdfontesError> {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(AdfontesError::Io {
                context: format!("inspecting stale staged file `{}`", path.display()),
                source,
            });
        }
    };
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("refusing to remove non-regular path `{}`", path.display()),
        });
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(AdfontesError::Io {
            context: format!("removing stale staged file `{}`", path.display()),
            source,
        }),
    }
}

struct ConfinedDirPlan {
    root: AllowedRoot,
    relative_tail: Vec<OsString>,
    expected_path: PathBuf,
}

fn plan_confined_staging_dir(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<ConfinedDirPlan, AdfontesError> {
    let raw_candidate = requested.map_or_else(
        || {
            policy
                .configured_staging_dir
                .clone()
                .unwrap_or_else(|| default_user_temp_subdir(policy))
        },
        Path::to_path_buf,
    );

    let abs_candidate = normalize_candidate_path(&raw_candidate)?;
    reject_forbidden_path(&abs_candidate, policy)?;

    let allowed_roots = collect_allowed_roots(policy)?;
    let matched = match_allowed_root(&abs_candidate, &allowed_roots, policy)?;
    let mut relative_tail = matched.tail;

    // Never write directly into shared world-writable `/tmp` or non-private temp root; use a
    // private per-UID `0700` subdirectory automatically.
    if relative_tail.is_empty() && matched.root.is_shared_temp {
        relative_tail.push(OsString::from(private_adfontes_subdir_name(policy)));
    }

    let expected_path = join_normal_segments(&matched.root.canonical, &relative_tail);
    reject_forbidden_path(&expected_path, policy)?;

    Ok(ConfinedDirPlan {
        root: matched.root.clone(),
        relative_tail,
        expected_path,
    })
}

struct MatchedRoot<'a> {
    root: &'a AllowedRoot,
    tail: Vec<OsString>,
}

#[derive(Debug, Clone)]
struct AllowedRoot {
    canonical: PathBuf,
    aliases: Vec<PathBuf>,
    is_shared_temp: bool,
    #[cfg(unix)]
    enforce_mode_on_root: bool,
}

fn collect_allowed_roots(policy: &StagingPolicy) -> Result<Vec<AllowedRoot>, AdfontesError> {
    let mut roots = Vec::new();
    let canon_system_tmp = dunce::canonicalize(&policy.system_tmp_dir).ok();
    if let Some(ref canonical) = canon_system_tmp {
        roots.push(AllowedRoot {
            canonical: canonical.clone(),
            aliases: unique_aliases(canonical, &normalize_lexical_only(&policy.system_tmp_dir)),
            is_shared_temp: true,
            #[cfg(unix)]
            enforce_mode_on_root: false,
        });
    }

    if !policy.temp_dir.is_absolute() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "system temp root `{}` must be an absolute path",
                policy.temp_dir.display()
            ),
        });
    }
    let abs_temp = normalize_candidate_path(&policy.temp_dir)?;
    reject_overbroad_temp_root(&abs_temp, canon_system_tmp.as_deref(), policy)?;
    reject_forbidden_path(&abs_temp, policy)?;
    if let Ok(canonical_temp) = dunce::canonicalize(&abs_temp) {
        reject_overbroad_temp_root(&canonical_temp, canon_system_tmp.as_deref(), policy)?;
        reject_forbidden_path(&canonical_temp, policy)?;
        let is_shared = is_temp_dir_shared(&canonical_temp, canon_system_tmp.as_deref(), policy);
        push_unique_root(
            &mut roots,
            AllowedRoot {
                aliases: unique_aliases(&canonical_temp, &abs_temp),
                canonical: canonical_temp,
                is_shared_temp: is_shared,
                #[cfg(unix)]
                enforce_mode_on_root: false,
            },
        );
    }

    for extra in &policy.extra_allowed_roots {
        if !extra.is_absolute() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "ADFONTES_EXTRA_ALLOWED_ROOTS entry `{}` must be an absolute path",
                    extra.display()
                ),
            });
        }
        let extra_root = validate_configured_staging_root(extra, policy, false)?;
        push_unique_root(&mut roots, extra_root);
    }

    if let Some(ref configured) = policy.configured_staging_dir {
        let configured_root = validate_configured_staging_root(configured, policy, true)?;
        push_unique_root(&mut roots, configured_root);
    }

    Ok(roots)
}

fn validate_configured_staging_root(
    configured: &Path,
    policy: &StagingPolicy,
    enforce_mode_on_root: bool,
) -> Result<AllowedRoot, AdfontesError> {
    #[cfg(not(unix))]
    let _ = enforce_mode_on_root;

    let abs_configured = normalize_candidate_path(configured)?;
    reject_overbroad_root(&abs_configured, policy)?;
    reject_forbidden_path(&abs_configured, policy)?;

    let canonical_configured = canonicalize_strict_existing_prefix(&abs_configured, policy)?;
    reject_overbroad_root(&canonical_configured, policy)?;
    reject_forbidden_path(&canonical_configured, policy)?;

    Ok(AllowedRoot {
        aliases: unique_aliases(&canonical_configured, &abs_configured),
        canonical: canonical_configured,
        is_shared_temp: false,
        #[cfg(unix)]
        enforce_mode_on_root,
    })
}

fn push_unique_root(roots: &mut Vec<AllowedRoot>, candidate: AllowedRoot) {
    if let Some(existing) = roots
        .iter_mut()
        .find(|root| root.canonical == candidate.canonical)
    {
        #[cfg(unix)]
        {
            existing.enforce_mode_on_root |= candidate.enforce_mode_on_root;
        }
        #[cfg(not(unix))]
        let _ = existing;
        return;
    }
    roots.push(candidate);
}

fn unique_aliases(canonical: &Path, secondary: &Path) -> Vec<PathBuf> {
    let mut aliases = vec![canonical.to_path_buf()];
    if canonical != secondary {
        aliases.push(secondary.to_path_buf());
    }
    aliases
}

fn match_allowed_root<'a>(
    abs_candidate: &Path,
    allowed_roots: &'a [AllowedRoot],
    policy: &StagingPolicy,
) -> Result<MatchedRoot<'a>, AdfontesError> {
    if let Some(matched) = find_best_root_match(abs_candidate, allowed_roots) {
        return Ok(matched);
    }

    #[cfg(windows)]
    {
        let canonical_candidate = if fs::symlink_metadata(abs_candidate)
            .is_ok_and(|meta| meta.is_file() && !is_symlink_or_reparse(&meta))
        {
            abs_candidate
                .parent()
                .zip(abs_candidate.file_name())
                .and_then(|(parent, file_name)| {
                    canonicalize_strict_existing_prefix(parent, policy)
                        .ok()
                        .map(|canon_parent| canon_parent.join(file_name))
                })
        } else {
            canonicalize_strict_existing_prefix(abs_candidate, policy).ok()
        };
        if let Some(canon_candidate) = canonical_candidate
            && let Some(matched) = find_best_root_match(&canon_candidate, allowed_roots)
        {
            return Ok(matched);
        }
    }

    #[cfg(not(windows))]
    let _ = policy;

    Err(AdfontesError::SafetyBlock {
        reason: format!(
            "path `{}` is outside allowed temp, ADFONTES_EXTRA_ALLOWED_ROOTS, and ADFONTES_STAGING_DIR roots",
            abs_candidate.display()
        ),
    })
}

fn find_best_root_match<'a>(
    candidate: &Path,
    allowed_roots: &'a [AllowedRoot],
) -> Option<MatchedRoot<'a>> {
    let mut best_match: Option<(&'a AllowedRoot, usize, Vec<OsString>)> = None;
    for root in allowed_roots {
        for alias in &root.aliases {
            if let Some(tail) = strip_allowed_alias_prefix(candidate, alias) {
                let depth = alias.components().count();
                if best_match
                    .as_ref()
                    .is_none_or(|(_, best_depth, _)| depth > *best_depth)
                {
                    best_match = Some((root, depth, tail));
                }
            }
        }
    }
    best_match.map(|(root, _, tail)| MatchedRoot { root, tail })
}

fn verify_existing_confined_path(
    matched_root: &AllowedRoot,
    tail_segments: &[OsString],
    expected_path: &Path,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    #[cfg(unix)]
    {
        let mut dir_file = match openat(CWD, &matched_root.canonical, DIR_OPEN_FLAGS, Mode::empty())
        {
            Ok(fd) => File::from(fd),
            Err(Errno::NOENT) => {
                let found =
                    find_existing_dir_ancestor(&matched_root.canonical, None, LABEL_STAGING_ROOT)?;
                if !is_os_root_or_system_tmp(&found.existing, policy) {
                    let ancestor_fd = openat_dir_fd(CWD, &found.existing)?;
                    verify_dir_fd_owner_and_mode(&ancestor_fd, &found.existing, policy, false)?;
                }
                return Ok(());
            }
            Err(errno) => return Err(map_openat_dir_error(errno, &matched_root.canonical)),
        };
        if !matched_root.is_shared_temp {
            verify_dir_fd_owner_and_mode(&dir_file, &matched_root.canonical, policy, false)?;
        }
        for segment in tail_segments {
            match openat(
                &dir_file,
                segment.as_os_str(),
                DIR_OPEN_FLAGS,
                Mode::empty(),
            ) {
                Ok(next_fd) => {
                    let next_file = File::from(next_fd);
                    verify_dir_fd_owner_and_mode(&next_file, expected_path, policy, false)?;
                    dir_file = next_file;
                }
                Err(Errno::NOENT) => return Ok(()),
                Err(errno) => return Err(map_openat_dir_error(errno, expected_path)),
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        let _ = policy;
        let mut current = matched_root.canonical.clone();
        for segment in tail_segments {
            current.push(segment);
            match fs::symlink_metadata(&current) {
                Ok(meta) => {
                    verify_non_unix_confined_dir_entry(
                        &current,
                        &meta,
                        &matched_root.canonical,
                        expected_path,
                    )?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(source) => {
                    return Err(AdfontesError::Io {
                        context: format!("statting staging directory `{}`", current.display()),
                        source,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(unix))]
fn verify_non_unix_no_symlink_ancestors(
    root: &Path,
    parent_dirs: &[OsString],
    display_path: &Path,
) -> Result<(), AdfontesError> {
    let mut current = root.to_path_buf();
    for segment in parent_dirs {
        current.push(segment);
        let meta = fs::symlink_metadata(&current).map_err(|source| AdfontesError::Io {
            context: format!("statting ancestor `{}`", current.display()),
            source,
        })?;
        verify_non_unix_confined_dir_entry(&current, &meta, root, display_path)?;
    }
    Ok(())
}

fn create_and_verify_confined_dir(
    matched_root: &AllowedRoot,
    tail_segments: &[OsString],
    expected_path: &Path,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    #[cfg(unix)]
    {
        let mut dir_file = open_or_create_root_unix(matched_root, policy)?;
        for segment in tail_segments {
            dir_file = mkdirat_and_verify_unix(&dir_file, segment, expected_path, policy, true)?;
        }
    }

    #[cfg(not(unix))]
    {
        let mut current = matched_root.canonical.clone();
        if !current.exists() {
            DirBuilder::new()
                .recursive(true)
                .create(&current)
                .map_err(|source| AdfontesError::Io {
                    context: format!("creating staging root `{}`", current.display()),
                    source,
                })?;
        }
        for segment in tail_segments {
            current.push(segment);
            match fs::symlink_metadata(&current) {
                Ok(meta) => {
                    verify_non_unix_confined_dir_entry(
                        &current,
                        &meta,
                        &matched_root.canonical,
                        expected_path,
                    )?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    DirBuilder::new()
                        .create(&current)
                        .map_err(|source| AdfontesError::Io {
                            context: format!("creating staging directory `{}`", current.display()),
                            source,
                        })?;
                }
                Err(source) => {
                    return Err(AdfontesError::Io {
                        context: format!("statting staging directory `{}`", current.display()),
                        source,
                    });
                }
            }
        }
    }

    finalize_canonical_confined_path(
        expected_path,
        &matched_root.canonical,
        policy,
        "canonicalizing staging directory",
        "canonical staging path",
    )
}

#[cfg(not(unix))]
fn verify_non_unix_confined_dir_entry(
    entry_path: &Path,
    meta: &fs::Metadata,
    allowed_root: &Path,
    display_path: &Path,
) -> Result<(), AdfontesError> {
    if is_symlink_or_reparse(meta) || !meta.is_dir() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "path `{}` traversed a symlink, junction, or non-directory component",
                display_path.display()
            ),
        });
    }
    let canonical = dunce::canonicalize(entry_path).map_err(|source| AdfontesError::Io {
        context: format!("canonicalizing directory `{}`", entry_path.display()),
        source,
    })?;
    if !path_starts_with_root(&canonical, allowed_root) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "path `{}` resolved outside allowed root `{}`",
                display_path.display(),
                allowed_root.display()
            ),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn open_or_create_root_unix(
    root: &AllowedRoot,
    policy: &StagingPolicy,
) -> Result<File, AdfontesError> {
    let found = find_existing_dir_ancestor(&root.canonical, None, LABEL_STAGING_ROOT)?;
    let mut cur_dir = openat_dir_fd(CWD, &found.existing)?;

    if found.missing_tail.is_empty() {
        if root.enforce_mode_on_root {
            verify_dir_fd_owner_and_mode(&cur_dir, &root.canonical, policy, true)?;
        } else if !root.is_shared_temp {
            verify_dir_fd_owner_and_mode(&cur_dir, &root.canonical, policy, false)?;
        }
        return Ok(cur_dir);
    }

    if !is_os_root_or_system_tmp(&found.existing, policy) {
        verify_dir_fd_owner_and_mode(&cur_dir, &found.existing, policy, false)?;
    }

    let total_missing = found.missing_tail.len();
    for (idx, segment) in found.missing_tail.into_iter().rev().enumerate() {
        let enforce_mode = idx + 1 == total_missing && root.enforce_mode_on_root;
        cur_dir =
            mkdirat_and_verify_unix(&cur_dir, &segment, &root.canonical, policy, enforce_mode)?;
    }

    Ok(cur_dir)
}

#[cfg(unix)]
fn open_confined_pdf_unix(
    matched_root: &AllowedRoot,
    parent_dirs: &[OsString],
    file_name: &OsString,
    display_path: &Path,
    policy: &StagingPolicy,
) -> Result<File, AdfontesError> {
    let mut cur_dir = openat_dir_fd(CWD, &matched_root.canonical)?;
    if !matched_root.is_shared_temp {
        verify_dir_fd_owner_and_mode(&cur_dir, &matched_root.canonical, policy, false)?;
    }

    for segment in parent_dirs {
        let next = openat(&cur_dir, segment.as_os_str(), DIR_OPEN_FLAGS, Mode::empty())
            .map(File::from)
            .map_err(|errno| map_openat_dir_error(errno, display_path))?;
        verify_dir_fd_owner_and_mode(&next, display_path, policy, false)?;
        cur_dir = next;
    }

    let file_flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    openat(&cur_dir, file_name.as_os_str(), file_flags, Mode::empty())
        .map(File::from)
        .map_err(|errno| {
            if matches!(errno, Errno::LOOP | Errno::ISDIR | Errno::NOTDIR) {
                AdfontesError::SafetyBlock {
                    reason: format!(
                        "PDF path `{}` must be a regular non-symlink file ({errno})",
                        display_path.display()
                    ),
                }
            } else {
                AdfontesError::Io {
                    context: format!("opening PDF `{}`", display_path.display()),
                    source: io::Error::from(errno),
                }
            }
        })
}

#[cfg(unix)]
fn openat_dir_fd<Fd: rustix::fd::AsFd>(dir_fd: Fd, path: &Path) -> Result<File, AdfontesError> {
    openat(dir_fd, path, DIR_OPEN_FLAGS, Mode::empty())
        .map(File::from)
        .map_err(|errno| map_openat_dir_error(errno, path))
}

#[cfg(unix)]
fn mkdirat_and_verify_unix(
    parent_fd: &File,
    segment: &OsStr,
    full_path: &Path,
    policy: &StagingPolicy,
    enforce_owner_only_mode: bool,
) -> Result<File, AdfontesError> {
    match mkdirat(parent_fd, segment, Mode::RWXU) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(errno) => {
            return Err(AdfontesError::Io {
                context: format!("creating staging directory `{}`", full_path.display()),
                source: io::Error::from(errno),
            });
        }
    }
    let next = openat(parent_fd, segment, DIR_OPEN_FLAGS, Mode::empty())
        .map(File::from)
        .map_err(|errno| map_openat_dir_error(errno, full_path))?;
    verify_dir_fd_owner_and_mode(&next, full_path, policy, enforce_owner_only_mode)?;
    Ok(next)
}

#[cfg(unix)]
fn verify_dir_fd_owner_and_mode(
    dir_file: &File,
    display_path: &Path,
    policy: &StagingPolicy,
    enforce_owner_only_mode: bool,
) -> Result<(), AdfontesError> {
    let meta = dir_file.metadata().map_err(|source| AdfontesError::Io {
        context: format!("statting staging directory `{}`", display_path.display()),
        source,
    })?;
    if !meta.is_dir() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "staging path `{}` is not a directory",
                display_path.display()
            ),
        });
    }
    if meta.uid() != policy.effective_uid {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "staging directory `{}` is owned by uid {}, expected uid {}",
                display_path.display(),
                meta.uid(),
                policy.effective_uid
            ),
        });
    }
    if enforce_owner_only_mode && meta.mode() & MODE_GROUP_OTHER_MASK != 0 {
        fchmod(dir_file, Mode::RWXU).map_err(|errno| AdfontesError::Io {
            context: format!(
                "restricting permissions on `{}` to 0700",
                display_path.display()
            ),
            source: io::Error::from(errno),
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn map_openat_dir_error(errno: Errno, path: &Path) -> AdfontesError {
    if matches!(errno, Errno::LOOP | Errno::NOTDIR | Errno::NOENT) {
        return AdfontesError::SafetyBlock {
            reason: format!(
                "staging path `{}` traversed a symlink or non-directory component ({errno})",
                path.display()
            ),
        };
    }
    AdfontesError::Io {
        context: format!("opening staging directory `{}`", path.display()),
        source: io::Error::from(errno),
    }
}

fn canonicalize_strict_existing_prefix(
    path: &Path,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    let found = find_existing_dir_ancestor(path, Some(policy), LABEL_STAGING_PATH)?;
    verify_no_intermediate_symlink_ancestors(&found.existing, policy)?;

    let mut canonical =
        dunce::canonicalize(&found.existing).map_err(|source| AdfontesError::Io {
            context: format!("canonicalizing ancestor `{}`", found.existing.display()),
            source,
        })?;
    for segment in found.missing_tail.into_iter().rev() {
        let mut comps = Path::new(&segment).components();
        match (comps.next(), comps.next()) {
            (Some(Component::Normal(normal)), None) => canonical.push(normal),
            _ => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "uncreated staging suffix in `{}` must contain only normal path segments",
                        path.display()
                    ),
                });
            }
        }
    }
    Ok(canonical)
}

struct ExistingDirAncestor {
    existing: PathBuf,
    missing_tail: Vec<OsString>,
}

fn find_existing_dir_ancestor(
    path: &Path,
    allow_alias_policy: Option<&StagingPolicy>,
    label: &str,
) -> Result<ExistingDirAncestor, AdfontesError> {
    let mut existing = path.to_path_buf();
    let mut missing_tail: Vec<OsString> = Vec::new();

    loop {
        match fs::symlink_metadata(&existing) {
            Ok(meta) => {
                let allowed_alias = allow_alias_policy
                    .is_some_and(|policy| is_known_system_alias(&existing, policy));
                if is_symlink_or_reparse(&meta) {
                    if !allowed_alias {
                        return Err(AdfontesError::SafetyBlock {
                            reason: format!(
                                "{label} ancestor `{}` must not be a symlink",
                                existing.display()
                            ),
                        });
                    }
                    if !fs::metadata(&existing).is_ok_and(|target| target.is_dir()) {
                        return Err(AdfontesError::SafetyBlock {
                            reason: format!(
                                "{label} ancestor `{}` exists and is not a directory",
                                existing.display()
                            ),
                        });
                    }
                } else if !meta.is_dir() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "{label} ancestor `{}` exists and is not a directory",
                            existing.display()
                        ),
                    });
                }
                return Ok(ExistingDirAncestor {
                    existing,
                    missing_tail,
                });
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let Some(name) = existing.file_name() else {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!("{label} `{}` has no existing ancestor", path.display()),
                    });
                };
                missing_tail.push(name.to_os_string());
                if !existing.pop() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!("{label} `{}` has no existing root", path.display()),
                    });
                }
            }
            Err(error) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "cannot inspect {label} ancestor `{}`: {error}",
                        existing.display()
                    ),
                });
            }
        }
    }
}

fn verify_no_intermediate_symlink_ancestors(
    existing: &Path,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    let mut current = PathBuf::new();
    for component in existing.components() {
        current.push(component.as_os_str());
        if matches!(
            component,
            Component::Prefix(_) | Component::RootDir | Component::CurDir
        ) {
            continue;
        }
        if is_known_system_alias(&current, policy) {
            continue;
        }
        let meta = fs::symlink_metadata(&current).map_err(|error| AdfontesError::SafetyBlock {
            reason: format!(
                "cannot inspect staging path ancestor `{}`: {error}",
                current.display()
            ),
        })?;
        if is_symlink_or_reparse(&meta) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "staging path ancestor `{}` must not be a symlink",
                    current.display()
                ),
            });
        }
        if !meta.is_dir() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "staging path ancestor `{}` exists and is not a directory",
                    current.display()
                ),
            });
        }
    }
    Ok(())
}

fn finalize_canonical_confined_path(
    path: &Path,
    allowed_root: &Path,
    policy: &StagingPolicy,
    io_context: &str,
    path_label: &str,
) -> Result<PathBuf, AdfontesError> {
    let verified = dunce::canonicalize(path).map_err(|source| AdfontesError::Io {
        context: format!("{io_context} `{}`", path.display()),
        source,
    })?;
    reject_forbidden_path(&verified, policy)?;
    ensure_within_root(&verified, allowed_root, path_label)?;
    Ok(verified)
}

fn ensure_within_root(
    verified: &Path,
    allowed_root: &Path,
    path_label: &str,
) -> Result<(), AdfontesError> {
    if !path_starts_with_root(verified, allowed_root) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "{path_label} `{}` escaped allowed root `{}`",
                verified.display(),
                allowed_root.display()
            ),
        });
    }
    Ok(())
}

fn reject_overbroad_temp_root(
    path: &Path,
    canon_system_tmp: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    if collect_normal_segments(path).is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "system temp root `{}` cannot be the filesystem root",
                path.display()
            ),
        });
    }
    let norm_system_tmp = normalize_lexical_only(&policy.system_tmp_dir);
    let matches_system_tmp = paths_equal_policy(path, &norm_system_tmp)
        || canon_system_tmp.is_some_and(|canon| paths_equal_policy(path, canon));
    if let Some(ref home) = policy.home_dir {
        for variant in collect_home_variants(home) {
            if paths_equal_policy(path, &variant)
                || (!matches_system_tmp && is_equal_or_ancestor_path(path, &variant))
            {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "system temp root `{}` cannot be the home directory or an ancestor of it",
                        path.display()
                    ),
                });
            }
        }
    }
    Ok(())
}

fn reject_overbroad_root(path: &Path, policy: &StagingPolicy) -> Result<(), AdfontesError> {
    if collect_normal_segments(path).is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "configured staging root `{}` cannot be the filesystem root",
                path.display()
            ),
        });
    }
    if let Some(ref home) = policy.home_dir {
        let home_variants = collect_home_variants(home);
        if home_variants
            .iter()
            .any(|home_variant| is_equal_or_ancestor_path(path, home_variant))
        {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "configured staging root `{}` cannot be the home directory or an ancestor of it",
                    path.display()
                ),
            });
        }
    }
    Ok(())
}

fn reject_forbidden_path(path: &Path, policy: &StagingPolicy) -> Result<(), AdfontesError> {
    let normal_segments = collect_normal_segments(path);

    for raw_prefix in &policy.forbidden_path_prefixes {
        if !raw_prefix.is_absolute() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "ADFONTES_FORBIDDEN_PATH_PREFIXES entry `{}` must be an absolute path",
                    raw_prefix.display()
                ),
            });
        }
        let norm_prefix = normalize_candidate_path(raw_prefix)?;
        if path.is_absolute() {
            let mut prefix_variants = vec![norm_prefix.clone()];
            if let Some(canon_prefix) = canonicalize_existing_prefix_best_effort(&norm_prefix)
                && !prefix_variants.contains(&canon_prefix)
            {
                prefix_variants.push(canon_prefix);
            }
            for prefix in &prefix_variants {
                let matched = {
                    #[cfg(windows)]
                    {
                        path_starts_with_root(path, prefix)
                    }
                    #[cfg(not(windows))]
                    {
                        let prefix_segments = collect_normal_segments(prefix);
                        prefix_segments.is_empty()
                            || segments_start_with_ignore_case(&normal_segments, &prefix_segments)
                    }
                };
                if matched {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging inside forbidden path prefix `{}` is blocked",
                            raw_prefix.display()
                        ),
                    });
                }
            }
        }
    }

    for segment in &normal_segments {
        if FORBIDDEN_PATH_SEGMENTS
            .iter()
            .any(|forbidden| segment.eq_ignore_ascii_case(forbidden))
        {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("staging inside sensitive directory `{segment}` is forbidden"),
            });
        }
    }

    if let Some(ref home) = policy.home_dir {
        for base in &collect_home_variants(home) {
            let base_segments = collect_normal_segments(base);
            if segments_start_with_ignore_case(&normal_segments, &base_segments) {
                let tail = normal_segments
                    .get(base_segments.len()..)
                    .unwrap_or_default();
                for forbidden_rel in FORBIDDEN_HOME_ROOTS {
                    if segments_start_with_ignore_case(tail, forbidden_rel) {
                        return Err(AdfontesError::SafetyBlock {
                            reason: format!(
                                "staging inside configuration root `~/{}` is forbidden",
                                forbidden_rel.join("/")
                            ),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

fn collect_home_variants(home: &Path) -> Vec<PathBuf> {
    let mut variants = vec![normalize_lexical_only(home)];
    if let Ok(canonical) = dunce::canonicalize(home)
        && !variants.contains(&canonical)
    {
        variants.push(canonical);
    }
    variants
}

fn normalize_candidate_path(path: &Path) -> Result<PathBuf, AdfontesError> {
    let raw_path = path.as_os_str();
    if raw_path.is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: "filesystem path must be non-empty".to_owned(),
        });
    }
    if raw_path.as_encoded_bytes().contains(&0)
        || path.to_string_lossy().chars().any(char::is_control)
    {
        return Err(AdfontesError::SafetyBlock {
            reason: "filesystem path must not contain NUL or control characters".to_owned(),
        });
    }
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "path `{}` must not contain `..` parent traversal segments",
                        path.display()
                    ),
                });
            }
            Component::Normal(segment) if has_unsafe_segment_syntax(segment) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "path `{}` contains unsafe verbatim traversal or separator syntax",
                        path.display()
                    ),
                });
            }
            _ => {}
        }
    }
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| AdfontesError::Io {
                context: "resolving current working directory".to_owned(),
                source,
            })?
            .join(path)
    };
    Ok(normalize_lexical_only(&abs))
}

fn canonicalize_existing_prefix_best_effort(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut tail_segments: Vec<OsString> = Vec::new();
    loop {
        if let Ok(mut canonical) = dunce::canonicalize(&existing) {
            for segment in tail_segments.into_iter().rev() {
                canonical.push(segment);
            }
            return Some(canonical);
        }
        let name = existing.file_name()?.to_os_string();
        tail_segments.push(name);
        if !existing.pop() {
            return None;
        }
    }
}

fn paths_equal_policy(lhs: &Path, rhs: &Path) -> bool {
    #[cfg(any(windows, target_os = "macos"))]
    {
        is_equal_or_ancestor_path(lhs, rhs) && is_equal_or_ancestor_path(rhs, lhs)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        lhs == rhs
    }
}

fn is_equal_or_ancestor_path(candidate: &Path, target: &Path) -> bool {
    #[cfg(windows)]
    {
        path_starts_with_root(target, candidate)
    }
    #[cfg(not(windows))]
    {
        if candidate.is_absolute() != target.is_absolute() {
            return false;
        }
        let candidate_segs = collect_normal_segments(candidate);
        let target_segs = collect_normal_segments(target);
        segments_start_with_ignore_case(&target_segs, &candidate_segs)
    }
}

fn is_symlink_or_reparse(meta: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        meta.file_type().is_symlink()
            || (meta.file_attributes() & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT) != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

fn is_known_system_alias(path: &Path, policy: &StagingPolicy) -> bool {
    let norm = normalize_lexical_only(path);
    if norm.as_os_str().is_empty() {
        return false;
    }
    [
        Some(normalize_lexical_only(&policy.system_tmp_dir)),
        Some(normalize_lexical_only(&policy.temp_dir)),
        policy
            .home_dir
            .as_ref()
            .map(|home| normalize_lexical_only(home)),
    ]
    .into_iter()
    .flatten()
    .any(|base| path_starts_with_root(&base, &norm))
}

fn path_starts_with_root(path: &Path, root: &Path) -> bool {
    strip_allowed_alias_prefix(path, root).is_some()
}

// `None` means `candidate` is outside `alias`, whereas `Some([])` means an exact match on `alias`.
fn strip_allowed_alias_prefix(candidate: &Path, alias: &Path) -> Option<Vec<OsString>> {
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let remainder = candidate.strip_prefix(alias).ok()?;
        let mut tail = Vec::new();
        for component in remainder.components() {
            match component {
                Component::Normal(segment) if !has_unsafe_segment_syntax(segment) => {
                    tail.push(segment.to_os_string());
                }
                Component::CurDir => {}
                _ => return None,
            }
        }
        Some(tail)
    }

    #[cfg(any(windows, target_os = "macos"))]
    {
        let simplified_candidate = dunce::simplified(candidate);
        let simplified_alias = dunce::simplified(alias);
        let candidate_parts: Vec<Component<'_>> = simplified_candidate
            .components()
            .filter(|comp| !matches!(comp, Component::CurDir))
            .collect();
        let alias_parts: Vec<Component<'_>> = simplified_alias
            .components()
            .filter(|comp| !matches!(comp, Component::CurDir))
            .collect();
        if candidate_parts.len() < alias_parts.len() {
            return None;
        }
        for (lhs, rhs) in candidate_parts.iter().zip(alias_parts.iter()) {
            if !components_equal_ignore_case(*lhs, *rhs) {
                return None;
            }
        }
        let mut tail = Vec::new();
        for component in candidate_parts.iter().skip(alias_parts.len()) {
            match component {
                Component::Normal(segment) if !has_unsafe_segment_syntax(segment) => {
                    tail.push(segment.to_os_string());
                }
                _ => return None,
            }
        }
        Some(tail)
    }
}

fn has_unsafe_segment_syntax(segment: &OsStr) -> bool {
    let bytes = segment.as_encoded_bytes();
    bytes == b"."
        || bytes == b".."
        || bytes.contains(&b'/')
        || (cfg!(windows) && bytes.contains(&b'\\'))
}

#[cfg(any(windows, target_os = "macos"))]
fn components_equal_ignore_case(lhs: Component<'_>, rhs: Component<'_>) -> bool {
    match (lhs, rhs) {
        #[cfg(windows)]
        (Component::Prefix(lhs_prefix), Component::Prefix(rhs_prefix)) => {
            windows_prefixes_match(lhs_prefix.kind(), rhs_prefix.kind())
        }
        (Component::RootDir, Component::RootDir) | (Component::CurDir, Component::CurDir) => true,
        (Component::Normal(left), Component::Normal(right)) => left
            .as_encoded_bytes()
            .eq_ignore_ascii_case(right.as_encoded_bytes()),
        _ => false,
    }
}

#[cfg(any(windows, test))]
fn windows_prefixes_match(lhs: Prefix<'_>, rhs: Prefix<'_>) -> bool {
    match (lhs, rhs) {
        (
            Prefix::Disk(left) | Prefix::VerbatimDisk(left),
            Prefix::Disk(right) | Prefix::VerbatimDisk(right),
        ) => left.eq_ignore_ascii_case(&right),
        (
            Prefix::UNC(left_server, left_share) | Prefix::VerbatimUNC(left_server, left_share),
            Prefix::UNC(right_server, right_share) | Prefix::VerbatimUNC(right_server, right_share),
        ) => {
            left_server
                .as_encoded_bytes()
                .eq_ignore_ascii_case(right_server.as_encoded_bytes())
                && left_share
                    .as_encoded_bytes()
                    .eq_ignore_ascii_case(right_share.as_encoded_bytes())
        }
        _ => lhs == rhs,
    }
}

fn segments_start_with_ignore_case<S: AsRef<str>>(segments: &[String], prefix: &[S]) -> bool {
    segments.len() >= prefix.len()
        && segments
            .iter()
            .zip(prefix.iter())
            .all(|(left, right)| left.eq_ignore_ascii_case(right.as_ref()))
}

fn collect_normal_segments(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect()
}

fn normalize_lexical_only(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in dunce::simplified(path).components() {
        if !matches!(component, Component::CurDir) {
            out.push(component.as_os_str());
        }
    }
    out
}

#[cfg(unix)]
fn is_os_root_or_system_tmp(path: &Path, policy: &StagingPolicy) -> bool {
    if path.components().count() <= 1 {
        return true;
    }
    dunce::canonicalize(&policy.system_tmp_dir).is_ok_and(|canon_sys| path == canon_sys)
}

fn is_temp_dir_shared(
    canonical_temp: &Path,
    canon_system_tmp: Option<&Path>,
    policy: &StagingPolicy,
) -> bool {
    if canon_system_tmp.is_some_and(|sys| sys == canonical_temp) {
        return true;
    }
    #[cfg(unix)]
    {
        fs::metadata(canonical_temp).is_ok_and(|meta| {
            meta.uid() != policy.effective_uid || meta.mode() & MODE_GROUP_OTHER_MASK != 0
        })
    }
    #[cfg(not(unix))]
    {
        let _ = policy;
        true
    }
}

fn join_normal_segments(base: &Path, segments: &[OsString]) -> PathBuf {
    let mut out = base.to_path_buf();
    for segment in segments {
        out.push(segment);
    }
    out
}

fn default_user_temp_subdir(policy: &StagingPolicy) -> PathBuf {
    policy.temp_dir.join(private_adfontes_subdir_name(policy))
}

fn private_adfontes_subdir_name(policy: &StagingPolicy) -> String {
    #[cfg(unix)]
    {
        format!("{DEFAULT_STAGING_NAME}-{}", policy.effective_uid)
    }
    #[cfg(not(unix))]
    {
        let _ = policy;
        DEFAULT_STAGING_NAME.to_owned()
    }
}

fn is_truthy_env(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Temporary staging directory wrapper for isolated unit and CLI tests.
#[cfg(test)]
pub struct TestSandbox {
    _dir: tempfile::TempDir,
    stage: PathBuf,
}

#[cfg(test)]
impl TestSandbox {
    /// Creates an isolated temporary sandbox directory for testing.
    ///
    /// # Panics
    ///
    /// Panics if the OS temporary directory or staging subdirectory cannot be created.
    #[must_use]
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary test sandbox must be creatable");
        let stage = dir.path().join("stage");
        fs::create_dir(&stage).expect("temporary test staging root must be creatable");
        Self { _dir: dir, stage }
    }

    /// Returns the filesystem path to the temporary sandbox root.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.stage
    }

    /// Constructs a [`StagingPolicy`] confined to this sandbox root.
    #[must_use]
    pub fn policy(&self) -> StagingPolicy {
        StagingPolicy::for_sandbox(self.path())
    }
}

#[cfg(test)]
impl Default for TestSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl StagingPolicy {
    pub(crate) fn for_sandbox(root: &Path) -> Self {
        Self {
            temp_dir: root.to_path_buf(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir: Some(root.with_file_name("home")),
            configured_staging_dir: None,
            extra_allowed_roots: Vec::new(),
            forbidden_path_prefixes: Vec::new(),
            disable_pdf_slice: false,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }

    pub(crate) fn write_test_pdf(
        &self,
        dir: &Path,
        file_name: &str,
        page_texts: &[&str],
    ) -> Result<PathBuf, AdfontesError> {
        let staging_dir = resolve_confined_staging_dir(Some(dir), self)?;
        let pdf_path = staging_dir.join(file_name);
        let bytes = crate::pdf::build_test_pdf(page_texts);
        write_file_atomic(&pdf_path, &bytes)?;
        Ok(pdf_path)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    #[cfg(unix)]
    use std::process::Command;

    use rstest::rstest;

    use super::*;
    #[cfg(unix)]
    use crate::pdf::build_test_pdf;

    struct SandboxHarness {
        sandbox: TestSandbox,
        policy: StagingPolicy,
    }

    impl SandboxHarness {
        fn new() -> Self {
            let sandbox = TestSandbox::new();
            let policy = sandbox.policy();
            Self { sandbox, policy }
        }

        fn path(&self) -> &Path {
            self.sandbox.path()
        }
    }

    fn test_policy(temp_root: &Path, home: &Path) -> StagingPolicy {
        StagingPolicy {
            temp_dir: temp_root.to_path_buf(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir: Some(home.to_path_buf()),
            configured_staging_dir: None,
            extra_allowed_roots: Vec::new(),
            forbidden_path_prefixes: Vec::new(),
            disable_pdf_slice: false,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }

    fn assert_unix_mode(path: &Path, expected_mode: u32) {
        #[cfg(unix)]
        {
            let mode = fs::metadata(path).unwrap().mode() & 0o777;
            assert_eq!(mode, expected_mode);
        }
        #[cfg(not(unix))]
        {
            let _ = (path, expected_mode);
        }
    }

    #[rstest]
    #[case("/mnt/workspace/src/user/repo")]
    #[case("//mnt/./workspace/src/user/repo")]
    #[case("/Volumes/workspace/src/user/repo")]
    #[case("/Volumes/WORKSPACE/src/user/repo")]
    #[case("/workspace_src/user/repo")]
    fn rejects_configured_forbidden_path_prefixes_case_insensitively(#[case] raw_path: &str) {
        let mut harness = SandboxHarness::new();
        harness.policy.forbidden_path_prefixes = vec![
            PathBuf::from("/mnt/workspace/src"),
            PathBuf::from("/Volumes/workspace/src"),
            PathBuf::from("/workspace_src"),
        ];
        assert!(matches!(
            resolve_confined_staging_dir(Some(Path::new(raw_path)), &harness.policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn rejects_relative_forbidden_prefix_and_relative_extra_allowed_root_fail_closed() {
        let harness = SandboxHarness::new();
        let valid_dir = harness.path().join("ok");
        let mut bad_forbidden = harness.policy.clone();
        bad_forbidden.forbidden_path_prefixes = vec![PathBuf::from("relative/workspace")];
        let mut bad_extra = harness.policy;
        bad_extra.extra_allowed_roots = vec![PathBuf::from("relative/scratch")];

        let forbidden_error =
            resolve_confined_staging_dir(Some(&valid_dir), &bad_forbidden).unwrap_err();
        let extra_error = resolve_confined_staging_dir(Some(&valid_dir), &bad_extra).unwrap_err();

        assert!(matches!(forbidden_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(extra_error, AdfontesError::SafetyBlock { .. }));
    }

    #[rstest]
    #[case(".gemini/config/skills")]
    #[case(".GEMINI/CONFIG/new")]
    #[case(".agents/skills")]
    #[case(".AGENTS/new")]
    #[case(".config/adfontes")]
    #[case(".local/bin")]
    #[case("repo/.git/hooks")]
    #[case("repo/.GIT/hooks")]
    #[case(".ssh")]
    #[case(".SSH")]
    fn rejects_config_and_sensitive_staging_paths(#[case] rel: &str) {
        let home = tempfile::tempdir().unwrap();
        let mut harness = SandboxHarness::new();
        harness.policy.home_dir = Some(home.path().to_path_buf());
        let forbidden = home.path().join(rel);
        harness.policy.configured_staging_dir = Some(forbidden.clone());
        assert!(matches!(
            resolve_confined_staging_dir(Some(&forbidden), &harness.policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn rejects_overbroad_configured_staging_root_and_control_chars() {
        let home = tempfile::tempdir().unwrap();
        let harness = SandboxHarness::new();
        let mut policy_root = test_policy(harness.path(), home.path());
        policy_root.configured_staging_dir = Some(PathBuf::from("/"));
        let mut policy_home = test_policy(harness.path(), home.path());
        policy_home.configured_staging_dir = Some(home.path().to_path_buf());
        let newline_dir = harness.path().join("bad\ndir");

        let root_failure = resolve_confined_staging_dir(None, &policy_root).unwrap_err();
        let home_failure = resolve_confined_staging_dir(None, &policy_home).unwrap_err();
        let newline_failure =
            resolve_confined_staging_dir(Some(&newline_dir), &harness.policy).unwrap_err();

        assert!(matches!(root_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(home_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(newline_failure, AdfontesError::SafetyBlock { .. }));
    }

    #[rstest]
    #[case(|p: &mut StagingPolicy, _: &Path| p.temp_dir = PathBuf::from("/"))]
    #[case(|p: &mut StagingPolicy, _: &Path| p.temp_dir = PathBuf::from("relative/tmp"))]
    #[case(|p: &mut StagingPolicy, root: &Path| p.temp_dir = root.join("../outside"))]
    #[case(|p: &mut StagingPolicy, root: &Path| p.home_dir = Some(root.to_path_buf()))]
    #[case(|p: &mut StagingPolicy, root: &Path| p.home_dir = Some(root.join("users/alice")))]
    #[case(|p: &mut StagingPolicy, root: &Path| p.temp_dir = root.join(".ssh"))]
    fn collect_allowed_roots_rejects_invalid_temp_dirs(
        #[case] configure: fn(&mut StagingPolicy, &Path),
    ) {
        let harness = SandboxHarness::new();
        let mut policy = harness.policy.clone();
        configure(&mut policy, harness.path());
        assert!(matches!(
            collect_allowed_roots(&policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn collect_allowed_roots_allows_sandbox_sibling_and_system_tmp_containing_home() {
        let harness = SandboxHarness::new();
        let sys_tmp = tempfile::tempdir().unwrap();
        let mut system_tmp_containing_home = harness.policy.clone();
        system_tmp_containing_home.system_tmp_dir = sys_tmp.path().to_path_buf();
        system_tmp_containing_home.temp_dir = sys_tmp.path().to_path_buf();
        system_tmp_containing_home.home_dir = Some(sys_tmp.path().join("runner/home"));

        assert!(collect_allowed_roots(&harness.policy).is_ok());
        assert!(collect_allowed_roots(&system_tmp_containing_home).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn configured_staging_dir_matching_extra_root_enforces_0700_mode() {
        use std::os::unix::fs::PermissionsExt;

        let mut harness = SandboxHarness::new();
        let shared_root = harness.path().join("shared-stage");
        fs::create_dir(&shared_root).unwrap();
        fs::set_permissions(&shared_root, fs::Permissions::from_mode(0o755)).unwrap();
        harness.policy.extra_allowed_roots = vec![shared_root.clone()];
        harness.policy.configured_staging_dir = Some(shared_root.clone());

        let resolved = resolve_confined_staging_dir(None, &harness.policy).unwrap();
        assert_eq!(resolved, dunce::canonicalize(&shared_root).unwrap());
        assert_unix_mode(&resolved, 0o700);
    }

    #[cfg(unix)]
    #[test]
    fn system_alias_symlink_to_regular_file_is_rejected() {
        let mut harness = SandboxHarness::new();
        let target_file = harness.path().join("regular-file");
        fs::write(&target_file, b"not a dir").unwrap();
        let home_link = harness.path().join("home-link");
        symlink(&target_file, &home_link).unwrap();
        harness.policy.home_dir = Some(home_link.clone());

        let failure = validate_staging_dir_policy(Some(&home_link.join("child")), &harness.policy)
            .unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn allows_extra_allowed_root_scratch_and_enforces_unix_0700_permissions() {
        let home = tempfile::tempdir().unwrap();
        let mut harness = SandboxHarness::new();
        harness.policy.home_dir = Some(home.path().to_path_buf());
        let extra_root = home.path().join(".agent-data/scratch");
        let agent_scratch = extra_root.join("conv-1/lit");
        fs::create_dir_all(&extra_root).unwrap();
        harness.policy.extra_allowed_roots = vec![extra_root];

        let resolved = resolve_confined_staging_dir(Some(&agent_scratch), &harness.policy).unwrap();
        assert!(resolved.is_dir());
        assert_unix_mode(&resolved, 0o700);
    }

    #[test]
    fn rejects_home_ancestor_extra_root_and_parent_dir_forbidden_prefix() {
        let outer = tempfile::tempdir().unwrap();
        let home_dir = outer.path().join("users/alice");
        fs::create_dir_all(&home_dir).unwrap();
        let harness = SandboxHarness::new();
        let mut ancestor_policy = test_policy(harness.path(), &home_dir);
        ancestor_policy.extra_allowed_roots = vec![outer.path().join("users")];

        let mut dotdot_policy = test_policy(harness.path(), &home_dir);
        dotdot_policy.forbidden_path_prefixes = vec![PathBuf::from("/mnt/foo/../secret")];

        let ancestor_error =
            resolve_confined_staging_dir(Some(&home_dir.join("sub")), &ancestor_policy)
                .unwrap_err();
        let dotdot_error =
            resolve_confined_staging_dir(Some(&harness.path().join("ok")), &dotdot_policy)
                .unwrap_err();

        assert!(matches!(ancestor_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(dotdot_error, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_forbidden_prefix_ancestor_and_symlinked_extra_root_ancestor() {
        let mut harness = SandboxHarness::new();
        let real_root = harness.path().join("real");
        let real_sub = real_root.join("sub");
        fs::create_dir_all(&real_sub).unwrap();
        let link_root = harness.path().join("link");
        symlink(&real_root, &link_root).unwrap();

        harness.policy.forbidden_path_prefixes = vec![link_root.join("future_blocked")];
        let mut extra_policy = harness.sandbox.policy();
        extra_policy.extra_allowed_roots = vec![link_root.join("sub")];

        let forbidden_error =
            validate_staging_dir_policy(Some(&real_root.join("future_blocked/x")), &harness.policy)
                .unwrap_err();
        let extra_error =
            validate_staging_dir_policy(Some(&real_sub.join("out")), &extra_policy).unwrap_err();

        assert!(matches!(forbidden_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(extra_error, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_live_dangling_symlinks_and_non_dir_ancestors() {
        let harness = SandboxHarness::new();
        let real_dir = harness.path().join("real");
        fs::create_dir(&real_dir).unwrap();
        let link_dir = harness.path().join("link");
        symlink(&real_dir, &link_dir).unwrap();
        let dangling = harness.path().join("dangling");
        symlink(harness.path().join("missing-target"), &dangling).unwrap();
        let regular_file = harness.path().join("regular");
        fs::write(&regular_file, b"not a dir").unwrap();

        let leaf_link = resolve_confined_staging_dir(Some(&link_dir), &harness.policy).unwrap_err();
        let mid_link = resolve_confined_staging_dir(Some(&link_dir.join("child")), &harness.policy)
            .unwrap_err();
        let dangling_mid =
            resolve_confined_staging_dir(Some(&dangling.join("child")), &harness.policy)
                .unwrap_err();
        let file_ancestor =
            resolve_confined_staging_dir(Some(&regular_file.join("sub")), &harness.policy)
                .unwrap_err();

        assert!(matches!(leaf_link, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(mid_link, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(dangling_mid, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(file_ancestor, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_foreign_uid_intermediate_directory_before_creating_child() {
        let mut harness = SandboxHarness::new();
        let shared = harness.path().join("foreign_parent");
        fs::create_dir(&shared).unwrap();
        harness.policy.effective_uid = harness.policy.effective_uid.wrapping_add(1);

        let failure =
            resolve_confined_staging_dir(Some(&shared.join("sub")), &harness.policy).unwrap_err();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
        assert!(!shared.join("sub").exists());
    }

    #[cfg(unix)]
    #[test]
    fn read_bounded_confined_pdf_blocks_unconfined_and_symlink_inputs() {
        let mut harness = SandboxHarness::new();
        let outside_dir = tempfile::tempdir().unwrap();
        harness.policy.system_tmp_dir = harness.path().join("fake-sys-tmp");

        let outside_pdf = outside_dir.path().join("secret.pdf");
        fs::write(&outside_pdf, build_test_pdf(&["Secret"])).unwrap();
        let real_pdf = harness.path().join("paper.pdf");
        fs::write(&real_pdf, build_test_pdf(&["Page 1"])).unwrap();
        let link_pdf = harness.path().join("link.pdf");
        symlink(&real_pdf, &link_pdf).unwrap();

        let outside_failure = read_bounded_confined_pdf(&outside_pdf, &harness.policy)
            .err()
            .unwrap();
        let link_failure = read_bounded_confined_pdf(&link_pdf, &harness.policy)
            .err()
            .unwrap();

        assert!(matches!(outside_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(link_failure, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn read_bounded_confined_pdf_rejects_fifo_without_blocking() {
        let harness = SandboxHarness::new();
        let fifo_path = harness.path().join("trap.pdf");
        let status = Command::new("mkfifo").arg(&fifo_path).status().unwrap();

        let failure = read_bounded_confined_pdf(&fifo_path, &harness.policy)
            .err()
            .unwrap();

        assert!(status.success());
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn validate_staging_dir_policy_rejects_existing_intermediate_symlink() {
        let harness = SandboxHarness::new();
        let target_dir = harness.path().join("real");
        let sub_dir = target_dir.join("nested");
        fs::create_dir_all(&sub_dir).unwrap();
        let link_dir = harness.path().join("link");
        symlink(&target_dir, &link_dir).unwrap();

        let failure = validate_staging_dir_policy(Some(&link_dir.join("nested")), &harness.policy)
            .unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn reject_overbroad_root_blocks_filesystem_root() {
        let harness = SandboxHarness::new();
        assert!(matches!(
            reject_overbroad_root(Path::new("/"), &harness.policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn windows_prefixes_match_handles_disk_and_unc_verbatim_equivalence() {
        assert!(windows_prefixes_match(
            Prefix::Disk(b'c'),
            Prefix::VerbatimDisk(b'C')
        ));
        assert!(!windows_prefixes_match(
            Prefix::Disk(b'c'),
            Prefix::VerbatimDisk(b'D')
        ));
        assert!(windows_prefixes_match(
            Prefix::UNC(OsStr::new("FileSrv"), OsStr::new("Papers")),
            Prefix::VerbatimUNC(OsStr::new("filesrv"), OsStr::new("papers"))
        ));
        assert!(!windows_prefixes_match(
            Prefix::UNC(OsStr::new("FileSrv"), OsStr::new("Papers")),
            Prefix::VerbatimUNC(OsStr::new("FileSrv"), OsStr::new("Other"))
        ));
    }

    #[rstest]
    #[case(".", true)]
    #[case("..", true)]
    #[case("sub/../../outside", true)]
    #[case("paper.pdf", false)]
    fn rejects_unsafe_verbatim_segment_syntax(#[case] raw: &str, #[case] expected: bool) {
        assert_eq!(has_unsafe_segment_syntax(OsStr::new(raw)), expected);
    }

    #[test]
    fn remove_file_if_present_deletes_regular_files_ignores_missing_and_blocks_directories() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.extracted.md");
        let regular = dir.path().join("stale.extracted.md");
        let subdir = dir.path().join("subdir");
        fs::write(&regular, b"old text").unwrap();
        fs::create_dir(&subdir).unwrap();

        assert!(remove_file_if_present(&missing).is_ok());
        assert!(remove_file_if_present(&regular).is_ok());
        assert!(!regular.exists());
        assert!(matches!(
            remove_file_if_present(&subdir).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[cfg(windows)]
    #[test]
    fn windows_strip_prefix_handles_verbatim_paths_and_blocks_traversal() {
        assert_eq!(
            strip_allowed_alias_prefix(
                Path::new(r"\\?\C:\Users\Alice\out\paper.pdf"),
                Path::new(r"c:\users\alice")
            ),
            Some(vec![OsString::from("out"), OsString::from("paper.pdf")])
        );
        assert!(
            strip_allowed_alias_prefix(
                Path::new(r"\\?\C:\Users\Alice\..\secret"),
                Path::new(r"C:\Users\Alice")
            )
            .is_none()
        );
        assert!(matches!(
            normalize_candidate_path(Path::new(r"\\?\C:\Users\Alice\sub/../secret")).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }
}
