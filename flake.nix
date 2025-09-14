{
  description = "edm";

  inputs = {
    nixpkgs.follows = "maipkgs/nixpkgs";
    maipkgs.url = "github:stephen-huan/maipkgs";
  };

  outputs = { self, nixpkgs, maipkgs }:
    let
      inherit (nixpkgs) lib;
      systems = lib.systems.flakeExposed;
      eachDefaultSystem = f: builtins.foldl' lib.attrsets.recursiveUpdate { }
        (map f systems);
    in
    eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (maipkgs.legacyPackages.${system}) python;
        python' = python.withPackages (ps: with ps; [
          numpy
          click
          pillow
          scipy
          torch
          psutil
          requests
          tqdm
          imageio
          imageio-ffmpeg
          # pyspng
        ]);
        formatters = [ pkgs.black pkgs.isort pkgs.nixpkgs-fmt ];
        linters = [ pkgs.pyright pkgs.ruff pkgs.statix ];
      in
      {
        devShells.${system}.default = (pkgs.mkShellNoCC.override {
          stdenv = pkgs.stdenvNoCC.override {
            initialPath = [ pkgs.coreutils ];
          };
        }) {
          packages = [
            python'
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
