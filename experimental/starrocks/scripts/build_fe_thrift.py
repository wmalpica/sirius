"""Build the FE's exact Thrift compiler without changing the StarRocks submodule.

Conda-forge skips some Thrift releases (including 0.23.0). Its compiler cannot
be used with an arbitrary Maven libthrift version: generated Java APIs change.
Only the compiler is built here, not Thrift's language runtimes or tests.
"""

import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


# SHA256 published with the Apache source release. Add a verified digest when
# updating the StarRocks submodule to a different Thrift runtime release.
SOURCE_SHA256 = {
    "0.23.0": "1859d932d2ae1f13d16c5a196931208c116310a5ff50f2bfd11d3db03be8f46f",
}


def main():
    workspace = Path(__file__).resolve().parents[1]
    pom = ET.parse(workspace / "starrocks/fe/pom.xml")
    version = pom.findtext("{*}properties/{*}thrift.version")
    if version not in SOURCE_SHA256:
        raise SystemExit(f"Add a verified Thrift source SHA256 for FE version {version!r}")

    thirdparty = workspace / ".pixi/starrocks-fe-thirdparty"
    compiler = thirdparty / "installed/bin/thrift"
    if compiler.is_file():
        actual = subprocess.check_output([compiler, "-version"], text=True).strip()
        if actual == f"Thrift version {version}":
            print(f"Using cached {actual}", flush=True)
            return

    thirdparty.mkdir(parents=True, exist_ok=True)
    archive = thirdparty / f"thrift-{version}.tar.gz"
    if not archive.is_file():
        url = f"https://archive.apache.org/dist/thrift/{version}/{archive.name}"
        print(f"Downloading {url}", flush=True)
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256[version]:
        raise SystemExit(f"SHA256 mismatch; remove {archive} and retry")

    source = thirdparty / f"thrift-{version}"
    if not source.is_dir():
        with tarfile.open(archive) as package:
            package.extractall(thirdparty, filter="data")
    build = thirdparty / f"thrift-{version}-build"
    subprocess.run(
        [
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_COMPILER=ON",
            "-DBUILD_LIBRARIES=OFF", "-DBUILD_TESTING=OFF",
            "-DBUILD_TUTORIALS=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "thrift-compiler", "--parallel", "4"],
        check=True,
    )
    compiler.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "compiler/cpp/bin/thrift", compiler)
    subprocess.run([compiler, "-version"], check=True)


if __name__ == "__main__":
    main()
