target("infer")
set_kind("static")
set_group("runtime")
add_options("with_dml")
-- deps
local packages = {"openmp","opencv","magic_enum"}
if has_config("with_dml") then
table.insert(packages,"directml")
table.insert(packages,"onnxruntime-dml")
else
table.insert(packages,"onnxruntime")
end
for _, v in ipairs(packages) do
    add_packages(v, { public = false })
end
add_files("*.cpp")
add_files("private/*.cpp")
add_headerfiles("*.h", { install = true })
add_headerfiles("*.hpp", { install = true })
add_headerfiles("private/*.h", { install = false })
add_headerfiles("private/*.hpp", { install = false })
add_includedirs(os.scriptdir(), { public = true })
add_includedirs("private", { public = false })
--links
add_syslinks("User32", "dxgi")
target_end()
--tests
for _,file in ipairs(os.files("test/test_*.cpp")) do
    local name = path.basename(file)
	target(name)
	set_kind("binary")
	set_default(false)
	set_group("runtime-tests")
	add_options("with_dml")
	for _, v in ipairs(packages) do
		add_packages(v, { public = false })
	end
	add_deps("infer")
	add_files(path.join("test",name..".cpp"))
	add_tests("default")
	after_build(function(target) 
		local t_pkgs = target:get("packages")
		for _,pkg_name in ipairs(t_pkgs) do
			local pkg = target:pkg(pkg_name)
			os.cp(pkg:installdir().."/bin/*.dll",target:targetdir().."/")
			os.cp(pkg:installdir().."/lib/*.dll",target:targetdir().."/")
		end
		os.cp("$(projectdir)/app/source/programs/demo/assets/",target:targetdir().."/")
	end)
	target_end()
end