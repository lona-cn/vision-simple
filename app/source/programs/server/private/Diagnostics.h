#pragma once
namespace vision_simple {
// Handles every nonempty argv without starting HTTP or initializing logging.
int RunDiagnosticsCLI(int argc, char* argv[]) noexcept;
}
