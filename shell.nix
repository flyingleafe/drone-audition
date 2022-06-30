with import <nixpkgs> {};

let 
  jaxlibSources = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html";
  pipenvWrapper = writeShellScriptBin "pipenvSource" ''
    PIP_FIND_LINKS=${jaxlibSources} ${pipenv}/bin/pipenv --python=${python39}/bin/python --site-packages $@
  '';

in stdenv.mkDerivation {
  name = "drone";
  buildInputs = [
    # System requirements.
    readline
    git
    vscodium

    python39
    python39Packages.pytorch
    python39Packages.numpy
    python39Packages.jax
    python39Packages.jaxlib

    pipenvWrapper
  ];
  src = null;
  shellHook = ''
    # Allow the use of wheels.
    SOURCE_DATE_EPOCH=$(date +%s)

    # Augment the dynamic linker path
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib.makeLibraryPath [stdenv.cc.cc libsndfile R readline]}
    
    pipenv () {
      source pipenvSource $@
    }

    pipenv shell
  '';
}
