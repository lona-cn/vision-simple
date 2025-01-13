IncludeSubDirs(os.scriptdir())
-- pkgs
add_requires("magic_enum 0.9.6","libhv 1.3.3","turbobase64","yalantinglibs","log4cplus")
-- onnxruntime
if has_config("with_dml") then
	add_requires("directml", { system = false })
	add_requires("onnxruntime-dml", { system = false })
elseif has_config("with_cuda") or has_config("with_tensorrt") then
	add_requires("onnxruntime", { system = false, configs = { gpu = true } })
else
	add_requires("onnxruntime", { system = false } )
end
-- opencv
if is_os("windows") then
	add_requires("opencv 4.10.0")
	add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
	protobuf = false } })
elseif is_os("linux") then
	add_requires("opencv 4.10.0")
	add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
	protobuf = false } })
	if is_arch("arm64") then
		add_requireconfs("libhv",{ configs = { defines = "PTHREAD_MUTEX_RECURSIVE", cxflags= "-Wno-implicit-function-declaration" } })
		add_requireconfs("yalantinglibs.cinatra",{configs = {aarch = true} })
		--add_requireconfs("yalantinglibs.async_simple",{ configs = { cxxflags="-Wno-error=unused-command-line-argument" } })
		add_requires("libffi",{system = true})
	end
end
