"""
===============================================================================
UrbanRoadDamage — Git Sync Utility
Script: scripts/git_sync.py
===============================================================================

PURPOSE:
Uses the local GitHub Desktop git binary and cached Windows Credential Manager
credentials to commit and push all recent updates (dataset additions, trained weights,
Gradio dynamic samples, and updated metrics) to GitHub.
===============================================================================
"""

import ctypes
from ctypes import wintypes
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_EXE = r"C:\Users\AAHNA\AppData\Local\GitHubDesktop\app-3.6.4\resources\app\git\cmd\git.exe"

# Windows Credential Manager structs
CRED_TYPE_GENERIC = 1

class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]

PCREDENTIAL = ctypes.POINTER(CREDENTIAL)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
CredReadW = advapi32.CredReadW
CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(PCREDENTIAL)]
CredReadW.restype = wintypes.BOOL
CredFree = advapi32.CredFree
CredFree.argtypes = [ctypes.c_void_p]


def get_github_token():
    target = "GitHub - https://api.github.com/AahnaUPES"
    p_cred = PCREDENTIAL()
    if CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(p_cred)):
        blob_size = p_cred.contents.CredentialBlobSize
        raw_bytes = ctypes.string_at(p_cred.contents.CredentialBlob, blob_size)
        CredFree(p_cred)
        # Windows Credential Manager stores UTF-16LE or ASCII/UTF-8
        try:
            return raw_bytes.decode("utf-16le").strip()
        except Exception:
            return raw_bytes.decode("utf-8", errors="ignore").strip()
    return None


def run_git_sync(commit_msg: str):
    print("=" * 70)
    print("URBAN ROAD DAMAGE — GIT COMMIT & PUSH TO GITHUB")
    print(f"Git binary : {GIT_EXE}")
    print(f"Repository : {REPO_ROOT}")
    print("=" * 70)

    # 1. Status
    res = subprocess.run([GIT_EXE, "status", "--short"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print("Modified / untracked files:")
    print(res.stdout or "(None)")

    # 2. Add
    print("\nStaging files...")
    subprocess.run([GIT_EXE, "add", "-A"], cwd=str(REPO_ROOT), check=True)

    # 3. Commit
    print(f"Committing with message: {commit_msg}")
    commit_res = subprocess.run([GIT_EXE, "commit", "-m", commit_msg], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(commit_res.stdout)
    if commit_res.stderr:
        print(commit_res.stderr)

    # 4. Push with token
    token = get_github_token()
    if not token:
        print("⚠️ GitHub token not found in Windows Credential Manager. Attempting normal push...")
        push_url = "origin"
    else:
        print("✅ Retrieved GitHub credentials from Windows Credential Manager.")
        push_url = f"https://AahnaUPES:{token}@github.com/AahnaUPES/urban-pothole-and-road-damage-assesstment-system-.git"

    print("Pushing to origin/main...")
    push_res = subprocess.run([GIT_EXE, "push", push_url, "main"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(push_res.stdout)
    if push_res.stderr:
        # Filter out token if in stderr
        safe_err = push_res.stderr.replace(token, "[REDACTED]") if token else push_res.stderr
        print(safe_err)

    if push_res.returncode == 0:
        print("\n🎉 Push to GitHub successfully completed!")
    else:
        print(f"⚠️ Push exited with return code: {push_res.returncode}")


if __name__ == "__main__":
    run_git_sync("feat: incorporate 15 user real-world road & pothole images, retrain detector, and update live Gradio demo")
