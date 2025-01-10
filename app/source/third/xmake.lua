IncludeSubDirs(os.scriptdir())
-- openmp magic_enum libhv
add_requires("openmp","magic_enum 0.9.6","libhv 1.3.3","turbobase64","yalantinglibs")
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
	if is_arch("arm64") then
		add_requires("opencv 4.6")
		add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,
		protobuf = false } })
	else
		add_requires("opencv",{ system = true })
	end
end
