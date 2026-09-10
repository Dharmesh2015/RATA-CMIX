# Building S1 with PGO + LTO + UPX on Google Cloud

`makefile` points here for PGO/LTO details; this document didn't exist before
and the reference was dangling. This is the real, tested procedure -- run
end to end on a fresh `n4d-highmem-2` / Ubuntu 20.04 focal VM -- including
the bugs that actually came up, not just the happy path.

## 1. Create the VM

Match the judging image as closely as possible:

```bash
gcloud compute instances create fast-vm-decomp \
  --zone=us-central1-a \
  --machine-type=n4d-highmem-2 \
  --boot-disk-size=100GB \
  --boot-disk-type=hyperdisk-balanced \
  --image=ubuntu-2004-focal-v20240731 \
  --image-project=ubuntu-os-cloud
```

**Zone stockouts are common and transient.** `n4d-highmem-2` is a popular
shape; a `ZONE_RESOURCE_POOL_EXHAUSTED` error just means try the next zone
in the same region (`-a`, `-b`, `-c`, `-f`) -- it is not a problem with the
command or the project. Whichever zone succeeds, use that same zone for
every command below.

Two warnings on create are expected and harmless:
- disk size (100 GB) larger than the base image (10 GB) -- cloud images
  auto-grow their root partition on boot
- the pinned image version is "deprecated" -- it still works; it is pinned
  specifically to match the judging environment, so do not switch to the
  suggested newer replacement

## 2. Get the source onto the VM

`git clone` over HTTPS fails non-interactively if the repo is private
(`fatal: could not read Username for 'https://github.com'`), and supplying
credentials to a remote VM is not something to script around. Instead,
ship the already-verified local tree directly:

```bash
# Locally:
git archive --format=tar -o rata_vm_upload.tar main
gcloud compute scp rata_vm_upload.tar fast-vm-decomp:rata_vm_upload.tar --zone=<ZONE>

# On the VM:
mkdir -p RATA-CMIX && tar -xf rata_vm_upload.tar -C RATA-CMIX && rm rata_vm_upload.tar
```

Do **not** pipe the archive through `gcloud compute ssh` directly
(`git archive | gcloud compute ssh ... "tar -x"`) -- on Windows, `gcloud`
shells out to PuTTY's `plink`, which does not pass binary stdin through
cleanly (`tar: This does not look like a tar archive`). The file-based
`scp` + extract above is binary-safe; verify with a hash if in doubt:

```bash
sha256sum models/6m-q4-fp32.tfwc5   # compare locally and on the VM
```

## 3. Install the toolchain

```bash
cd RATA-CMIX
sudo bash install.sh   # not ./install.sh -- see the exec-bit note below
```

**Every `.sh` script in this repo was committed without the executable
bit** (mode `100644`, not `100755`) until this was fixed directly in git
with `git update-index --chmod=+x <file>` for all of `install.sh`,
`build.sh`, `build_and_construct_comp.sh`, `install_tools/*.sh`, and
`tools/*.sh`. On a checkout from before that fix, `./install.sh` fails
with `command not found` (not `permission denied` -- that is the
characteristic symptom); `bash install.sh` works regardless of the bit.

Confirm the toolchain landed correctly:

```bash
clang++-17 --version
/opt/upx/upx-5.1.1-amd64_linux/upx --version   # or plain `upx --version`
```

## 4. Build S1 with the existing profile (works today)

```bash
bash build_and_construct_comp.sh
```

This is the documented, working path: `make clean && make target93` (LTO
`-flto=thin` + whatever `pgo/default.profdata` is already committed),
strip, UPX `--ultra-brute` (verified with `upx -t`), then a real
compress/decompress self-check of `dictionary/english.dic` and the
article-order file before appending them plus
`models/6m-q4-fp32.tfwc5` into the final `cmix` binary.

Expect `-Wbackend-plugin` "function control flow change detected (hash
mismatch)" warnings if the committed profile was trained on a different
clang++-17 build than this VM's -- harmless, LLVM falls back to default
heuristics for just those functions, never a build failure. This is the
`pgo/default.profdata` staleness the README already documents.

## 5. Regenerating a fresh, toolchain-matched profile

This is where it gets real. Two genuine bugs surfaced doing this on a
completely fresh VM; both are documented here because the failure modes
are silent or misleading otherwise.

### 5a. `make pgo-instrumented` crashes the linker

```
ld.lld: warning: .../libclang_rt.profile-x86_64.a(...): --icf=safe
  conservatively ignores SHT_LLVM_ADDRSIG ...
clang++-17: error: unable to execute command: Segmentation fault (core dumped)
make: *** [makefile:181: pgo-instrumented] Error 2
```

Root cause: `LDFLAGS` unconditionally includes `-Wl,--icf=safe`
(identical code folding), and this specific LLVM 17.0.6 build's `ld.lld`
segfaults combining `--icf=safe` with ThinLTO **and** the profiling
runtime (`libclang_rt.profile-x86_64.a`, only linked for
`-fprofile-generate` builds). The real production build (`target93`,
`-fprofile-use`, no profiling runtime linked) is unaffected -- confirmed
directly by building it standalone before touching PGO regeneration at
all.

Workaround: drop `--icf=safe` for the instrumented build only, by
overriding `LDFLAGS` on the command line (GNU Make command-line
assignments take precedence over the makefile's own `:=`, and propagate
into the recipe's recursive `$(MAKE)` call):

```bash
make pgo-instrumented LDFLAGS='-m64 -fuse-ld=lld -Wl,--gc-sections -std=c++17 -flto=thin -fprofile-generate=pgo-raw'
```

This links cleanly. `--icf=safe` stays intact for the real `target93`
build in step 4/6 -- only the instrumented probe drops it.

### 5b. The documented `-e prof_input/input` recipe does not work

The README (and this file's own prior non-existence) points at:

```bash
./cmix_pgo_instrumented -e prof_input/input   profile_out_1
./cmix_pgo_instrumented -e prof_input/input2  profile_out_2
make pgo-merge
```

**This does not currently work**, for a structural reason, not a flag or
environment issue:

- `runner.cpp`'s `-e` path is `selfextract_comp()` -> `split4Comp()` ->
  `reorder()` -> `phda9_prepr()` -> compress.
- `selfextract_comp()` does `fopen("cmix", "rb")` with **no null check**
  before `fseek`ing it -- if no file literally named `cmix` exists in the
  working directory, this segfaults immediately (`fseek + 23`, fault
  address `0x0`). A genuinely-packaged `cmix` (from step 4) must already
  exist in the same directory before running `-e` mode at all, under
  *any* binary name.
- Even with a real `cmix` present, `reorder()` uses the article-order data
  embedded in it -- 1,094,862 bytes of ordering instructions indexed
  against the **real, complete** enwik9's actual article positions.
  `prof_input/input` (50,051 bytes) and `prof_input/input2` (930,723
  bytes) are small samples, not the real enwik9, so those indices do not
  correspond to anything in them. The pipeline runs every stage --
  `split4Comp` produces `.main`/`.intro`/`.coda`, `.dict` and
  `.new_article_order` get restored correctly -- and then simply produces
  no output and exits 0, with no error at all. This is easy to mistake
  for "it's still working."
- **Do not substitute a placeholder/bogus `cmix`** to get past the fopen
  check (e.g. `cp cmix_pgo_instrumented cmix`). With no genuine embedded
  dictionary/article-order data, `-e` mode hits
  `cmix error: .new_article_order.comp was compressed with a dictionary,
  but none was provided` and then enters a multi-day death spiral --
  observed directly as `remaining: 101:42:33` and climbing. This is the
  same failure class documented separately for ALTXS: `-e` mode requires
  the genuine, structurally complete enwik9 article set and cannot be
  shortcut with a smaller or synthetic stand-in. Kill it
  (`pkill -9 -f cmix_pgo_instrumented`, then check for and kill the
  orphaned `./cmix -d .dict.comp .dict` child it spawns) and clean up
  `ppm.temp` (it will have grown to ~14.68 GB), `.dict`,
  `.new_article_order`, `.main*`, `.intro`, `.coda`, and any partial
  output before retrying anything.

**Net effect: there is currently no fast path to regenerate
`pgo/default.profdata` against a specific toolchain.** The only input
`-e` mode accepts is the real, complete enwik9, which puts profile
regeneration on the same multi-day timescale as the full compression run
itself -- defeating the purpose of a quick, VM-local profile refresh.
Until `-e` mode (or a dedicated profiling entry point) can accept a
smaller representative sample, building S1 with the existing committed
profile (step 4) is the practical path, and the resulting hash-mismatch
warnings are the documented, harmless cost of that.

## 6. Rebuild S1 for real

If step 5 ever does produce a fresh `pgo/default.profdata` (`make
pgo-merge`), rebuild the real artifact against it:

```bash
bash build_and_construct_comp.sh
```

This overwrites `cmix` in the repo root with the freshly-profiled build.
Otherwise, the `cmix` from step 4 already **is** S1 -- nothing further to
do.

## 7. Download S1

```bash
sha256sum cmix   # note this down before leaving the VM
gcloud compute scp fast-vm-decomp:RATA-CMIX/cmix "<local submission path>" --zone=<ZONE>
sha256sum "<local submission path>"   # must match exactly
```

**Watch the destination path.** `gcloud compute scp` on Windows (via
`pscp`) has been observed landing a file in an unexpected subdirectory
even when a full destination file path was given, rather than exactly
where specified. Always verify the file actually landed where intended
(`ls`) and that its hash matches, and move it into place if not -- don't
assume the scp destination argument was honored literally.

## 8. Clean up

```bash
gcloud compute instances delete fast-vm-decomp --zone=<ZONE> --quiet
gcloud compute instances list   # confirm 0 items -- billing has stopped
```

The VM bills for the entire time it exists, not just while a build is
running -- delete it as soon as the artifact is downloaded and verified,
not at the end of a longer session.

## Reference: a completed run

One full run of steps 1-4 and 7-8 (existing committed profile, no fresh
regeneration) measured:

| Component | Bytes |
| --- | ---: |
| Packed core (UPX `--ultra-brute`) | 166,076 |
| Embedded dictionary | 100,054 |
| Embedded article order | 200,834 |
| Transformer weights (`6m-q4-fp32.tfwc5`) | 2,902,452 |
| **Hutter S1** | **3,369,432** |

Build time: a few minutes end to end (toolchain already installed), not
counting the VM boot itself. This is a code-size measurement of one build,
not a compression benchmark -- no claim about `archive9`/S2 or a judged
score is made or implied here.
