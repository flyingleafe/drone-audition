with import <nixpkgs> {};

let 
  pipenvWrapper = writeShellScriptBin "pipenvSource" ''
    ${pipenv}/bin/pipenv --python=${python310}/bin/python --site-packages $@
  '';

in stdenv.mkDerivation {
  name = "drone";
  buildInputs = [
    # System requirements.
    readline
    git
    vscodium

    python310
    python310Packages.pytorch
    python310Packages.numpy
    python310Packages.jax
    python310Packages.jaxlib

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
