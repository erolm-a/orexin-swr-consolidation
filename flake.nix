# Impure UV workflow. It downloads standard Python and all dependencies from pypa.
# All architectures are supported, but some get extra cuda support.

{
  description = "Dingo project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    # nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      # https://nixcademy.com/posts/new-default-nix-flake-template/
      inherit (nixpkgs) lib;

      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      cudaSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      eachSystem = systems: f: builtins.foldl' (a: s: a // { ${s} = f s; }) { } systems;

      makeShell =
        { system, enableCuda }:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = enableCuda;
            };
          };

          cudaPkgs = if enableCuda then pkgs.cudaPackages_12_8 else [ ];

          cudaLibs =
            if enableCuda then
              with cudaPkgs;
              [
                cudnn
                nccl
                libcutensor
                libcublas
                libcusparse
                libcusparse_lt
                libcusolver
                libcurand
                cudatoolkit
              ]
            else
              [ ];

          qtPlatformLibs =
            if pkgs.stdenv.isLinux then
              with pkgs;
              [
                wayland
                libxkbcommon
                fontconfig
                libxcb
                libxkbcommon
                freetype
                harfbuzz
                zstd
                dbus
              ]
            else
              with pkgs;
              [
                fontconfig
                freetype
                harfbuzz
                zstd
              ];

          jvm = pkgs.jdk21_headless;

          packages = with pkgs; [
            uv
            qt6.qtbase
            qt6.wrapQtAppsHook
          ];

          nativeBuildInputs = with pkgs; [
            cacert
          ];

          env = {
            # Prevent uv from managing Python downloads
            #UV_PYTHON_DOWNLOADS = "never";
            # Force uv to use nixpkgs Python interpreter
            #UV_PYTHON = "${wrappedPython}/bin/python";

            IPYTHONDIR = ".ipython";

            SHELL = "${pkgs.zsh}/bin/zsh";
            XLA_PYTHON_CLIENT_PREALLOCATE = "false";
            CC = "${pkgs.clang}/bin/clang";
            CXX = "${pkgs.clang}/bin/clang++";
            JAX_COMPILATION_CACHE_DIR = "./.jaxcache";
            LINE_PROFILE = 1;
            USE_OPENMP = "TRUE";
            OPENMP_NUM_THREADS = "8";
          }
          // lib.optionalAttrs pkgs.stdenv.isLinux {
            # Python libraries often load native shared objects using dlopen(3).
            # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.

            NIX_LD_LIBRARY_PATH =
              with pkgs;
              # It is impossible to have a pure flake when using nvidia. At least with (moderate) driver updates it shouldn't be necessary to update this, unless your cuda drivers also change
              "/run/opengl-driver/lib:"
              + lib.makeLibraryPath (cudaLibs ++ qtPlatformLibs ++ pythonManylinuxPackages.manylinux1);
            NIX_LD = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

            XLA_FLAGS =
              (if enableCuda then "--xla_gpu_cuda_data_dir=${cudaPkgs.cudatoolkit}" else "")
              + "--xla_force_host_platform_device_count=8";
          }
          // lib.optionalAttrs (pkgs.stdenv.isDarwin) {
            # macOS uses DYLD_LIBRARY_PATH and framework paths
            DYLD_LIBRARY_PATH = lib.makeLibraryPath (qtPlatformLibs);
            QT_QPA_PLATFORM = "cocoa";
          };
          shellHook = ''
            unset PYTHONPATH

            if [[ ! -d .venv ]]; then uv venv --python 3.12; fi  # specify python version
            source .venv/bin/activate

          '';

        in
        pkgs.mkShell {
          inherit packages;
          inherit env;
          inherit shellHook;
          inherit nativeBuildInputs;
        };

    in
    {
      devShells = (
        eachSystem allSystems (
          system:
          {
            default = makeShell {
              inherit system;
              enableCuda = false;
            };
          }
          // lib.optionalAttrs (lib.elem system cudaSystems) {
            cuda = makeShell {
              inherit system;
              enableCuda = true;
            };
          }
        )
      );
    };
}
