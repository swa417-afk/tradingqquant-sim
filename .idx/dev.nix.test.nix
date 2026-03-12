
# .idx/dev.nix.test.nix
#
# This file contains tests for the .idx/dev.nix configuration.
# To run these tests, you can use the following command in your terminal:
#
# nix-instantiate --eval --strict .idx/dev.nix.test.nix
#
# A successful evaluation will output 'true'. Any errors indicate a failed test.

let
  pkgs = import <nixpkgs> {};
  # The configuration from dev.nix
  config = import ./dev.nix { inherit pkgs; };

  # Helper for running tests
  runTest = name: condition:
    if condition
    then true
    else throw "Test failed: ${name}";

  # Test suite
  tests = [
    (runTest "Channel is stable-24.05" (config.channel == "stable-24.05"))
    (runTest "python311 is in packages" (pkgs.lib.elem pkgs.python311 config.packages))
    (runTest "pip is in packages" (pkgs.lib.elem pkgs.python311Packages.pip config.packages))
    (runTest "anthropic is in packages" (pkgs.lib.elem pkgs.python311Packages.anthropic config.packages))
    (runTest "Gemini extension is present" (pkgs.lib.elem "google.gemini-cli-vscode-ide-companion" config.idx.extensions))
    (runTest "Previews are enabled" (config.idx.previews.enable == true))
    (runTest "Default files to open are correct" (
      pkgs.lib.elem ".idx/dev.nix" config.idx.workspace.onCreate.default.openFiles &&
      pkgs.lib.elem "README.md" config.idx.workspace.onCreate.default.openFiles
    ))
    (runTest "API_KEY is set" (config.env.API_KEY == "your-secret-key"))
    (runTest "Build task is correct" (config.idx.workspace.onCreate.build == "touch build.log"))
  ];

in
  # If all tests pass, this will evaluate to true
  pkgs.lib.foldl' (acc: test: acc && test) true tests
