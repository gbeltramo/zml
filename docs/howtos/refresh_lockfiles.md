# Refreshing ZML's Debian Package Lockfiles

## Overview

ZML uses Bazel to build its runtime and compiler components, but some of its platform dependencies are distributed as Debian (`.deb`) packages.

Those packages are not installed using the host system's `apt` package manager. Instead, ZML maintains Bazel-managed package lockfiles containing the exact package versions, download URLs, SHA-256 checksums, architectures, and package dependencies required by each platform.

For the CPU platform, the relevant files are:

```text
platforms/cpu/packages.yaml
platforms/cpu/packages.lock.json
```

The package manifest contains the packages ZML requires:

```yaml
version: 1

sources:
  - channel: bookworm main
    url: https://snapshot-cloudflare.debian.org/archive/debian/20250711T030400Z
  - channel: llvm-toolchain-bookworm-23 main
    url: https://apt.llvm.org/bookworm

archs:
  - "amd64"

packages:
  - "llvm-libunwind1"
```

At the top of the manifest, ZML documents the command used to generate the lockfile `bazel run @apt_cpu//:lock`

This command was the appropriate fix for the failed `llvm-libunwind1` download described below.

## The original failure

The ZML build initially failed while Bazel was fetching the CPU plugin's `llvm-libunwind1` dependency.

The important part of the error was:

```text
Error downloading [
  https://apt.llvm.org/bookworm/pool/main/l/llvm-toolchain-23/llvm-libunwind1_23.1.2~++20260916013112+21a77e7bb6bf-1~exp1~20260916133123.74_amd64.deb
]:
GET returned 404 Not Found
```

The dependency came from `platforms/cpu/packages.lock.json`

The lockfile contained an entry similar to:

```json
{
    "arch": "amd64",
    "key": "llvm-libunwind1_1-23.1.2_-p--p-20260916013112-p-21a77e7bb6bf-1_exp1_20260916133123.74_amd64",
    "name": "llvm-libunwind1",
    "sha256": "7fe1e2f61f6db688b53a4015a2d391f4474e0c0aec76b4dc262fe91a43854e84",
    "urls": [
        "https://apt.llvm.org/bookworm/pool/main/l/llvm-toolchain-23/llvm-libunwind1_23.1.2~++20260916013112+21a77e7bb6bf-1~exp1~20260916133123.74_amd64.deb"
    ],
    "version": "1:23.1.2~++20260916013112+21a77e7bb6bf-1~exp1~20260916133123.74"
}
```

The problem was that the exact `.deb` filename recorded in ZML's lockfile was no longer available from apt.llvm.org.

## What bazel run @apt_cpu//:lock does

The command is `bazel run @apt_cpu//:lock`

`@apt_cpu` is a Bazel-managed external repository, and `:lock` is a Bazel target provided for generating the package lock information.

The command uses the package manifest associated with the CPU platform `platforms/cpu/packages.yaml`/

That manifest tells the package resolver:

- Which Debian repositories to use.
- Which LLVM repository to use.
- Which architectures to resolve.
- Which packages are required.

For the CPU platform, those are:

```yaml
sources:
  - channel: bookworm main
    url: https://snapshot-cloudflare.debian.org/archive/debian/20250711T030400Z

  - channel: llvm-toolchain-bookworm-23 main
    url: https://apt.llvm.org/bookworm

archs:
  - "amd64"

packages:
  - "llvm-libunwind1"
```

The resolver then determines the package metadata needed to create a reproducible lockfile.

The resulting information includes fields such as:

```json
{
    "arch": "amd64",
    "name": "llvm-libunwind1",
    "version": "...",
    "sha256": "...",
    "urls": ["..."],
    "dependencies": [...]
}
```

That information is written to `platforms/cpu/packages.lock.json`
