set_xmakever("2.9.4")
set_project("vision-simple")

-- language version
set_languages("c23", "c++23")
-- compiler float mode
set_fpmodels("fast")
-- basic
if is_mode("release") then
    set_optimize("fastest")
    set_runtimes("MT")
    set_strip("all")
    set_warnings("none")
    add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "f16c", "avx", "avx2")
    set_symbols("hidden")
elseif is_mode("releasedbg") then
    set_optimize("none")
    set_runtimes("MDd")
    add_defines("VISION_SIMPLE_DEBUG")
    set_symbols("debug")
    set_warnings("all")
    --set_warnings("all", "error")
end
-- platform flags
if is_plat("windows") then
	add_ldflags("/LTCG")
	add_defines(
			"_CRT_SECURE_NO_WARNINGS",
			"_UNICODE",
			"UNICODE",
			"_CONSOLE",
			"NOMINMAX",
			"NOGDI",
			"WIN32_LEAN_AND_MEAN",
			"_WIN32_WINNT=0x0A00"
	)
	add_cxxflags("/d1trimfile:$(curdir)\\")
	add_cxxflags("/experimental:deterministic")
	add_ldflags("/PDBALTPATH:%_PDB%")
	add_cxxflags("/Zc:preprocessor")
	add_cxxflags("/Zc:__cplusplus")
	add_cxxflags("/utf-8")
	add_ldflags("/MAP")
	if is_mode("releasedbg") then
		add_defines("_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH")
	end
end
includes("xmake/setup.lua")