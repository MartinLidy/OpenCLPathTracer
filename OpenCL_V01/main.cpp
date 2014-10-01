#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "objLoader.h"
#include "obj_parser.h"

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

static cl_command_queue command_queue = NULL;


struct vector3d
{
	float X, Y, Z;

	inline vector3d(void) {}
	inline vector3d(const float x, const float y, const float z)
	{
		X = x; Y = y; Z = z;
	}

	inline vector3d operator + (const vector3d& A) const
	{
		return vector3d(X + A.X, Y + A.Y, Z + A.Z);
	}

	inline vector3d operator + (const float A) const
	{
		return vector3d(X + A, Y + A, Z + A);
	}

	inline float Dot(const vector3d& A) const
	{
		return A.X*X + A.Y*Y + A.Z*Z;
	}
};


struct Image
{
	std::vector<char> pixel;
	int width, height;
};

void printVector(obj_vector *v)
{
	printf("%.2f,", v->e[0]);
	printf("%.2f,", v->e[1]);
	printf("%.2f  ", v->e[2]);
}


Image LoadImage (const char* path)
{
	std::ifstream in (path, std::ios::binary);

	std::string s;
	in >> s;

	if (s != "P6") {
		exit (1);
	}

	// Skip comments
	for (;;) {
		getline (in, s);

		if (s.empty ()) {
			continue;
		}

		if (s [0] != '#') {
			break;
		}
	}

	std::stringstream str (s);
	int width, height, maxColor;
	str >> width >> height;
	in >> maxColor;

	if (maxColor != 255) {
		exit (1);
	}

	{
		// Skip until end of line
		std::string tmp;
		getline(in, tmp);
	}

	std::vector<char> data (width * height * 3);
	in.read (reinterpret_cast<char*> (data.data ()), data.size ());

	const Image img = { data, width, height };
	return img;
}

void SaveImage (const Image& img, const char* path)
{
	std::ofstream out (path, std::ios::binary);

	out << "P6\n";
	out << img.width << " " << img.height << "\n";
	out << "255\n";
	out.write (img.pixel.data (), img.pixel.size ());
}

Image RGBtoRGBA (const Image& input)
{
	Image result;
	result.width = input.width;
	result.height = input.height;

	for (std::size_t i = 0; i < input.pixel.size (); i += 3) {
		result.pixel.push_back (input.pixel [i + 0]);
		result.pixel.push_back (input.pixel [i + 1]);
		result.pixel.push_back (input.pixel [i + 2]);
		result.pixel.push_back (0);
	}

	return result;
}

Image RGBAtoRGB (const Image& input)
{
	Image result;
	result.width = input.width;
	result.height = input.height;

	for (std::size_t i = 0; i < input.pixel.size (); i += 4) {
		result.pixel.push_back (input.pixel [i + 0]);
		result.pixel.push_back (input.pixel [i + 1]);
		result.pixel.push_back (input.pixel [i + 2]);
	}

	return result;
}

std::string GetPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string GetDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram (const std::string& source,
	cl_context context)
{
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateProgramWithSource.html
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

float *VertsToFloat3(obj_vector** verts, int vertCount){
	int arraySize = vertCount * 3;
	float *output = (float*)malloc(arraySize * sizeof(float));

	// Each vert
	for (int i = 0; i < vertCount; i++){
		printf("Vert: %d = [", i);
		// Each coord
		for (int k = 0; k < 3; k++){
			output[i * 3 + k] = verts[i]->e[k];
			printf("%f, ", output[i * 3 + k]);
		}
		printf("]\n");
	}

	return output;
}

int *FacesToVerts(obj_face** faces, int faceCount){
	int arraySize = faceCount * 3;
	int *output = (int*)malloc(arraySize * sizeof(int));
	
	// Each face
	for (int i = 0; i < faceCount; i++){
		printf("Face: %d\n", i);
		// Each vert
		for (int k = 0; k < 3; k++){
			output[i * 3 + k] = faces[i]->vertex_index[k];
			printf("   Vert: %d", output[i * 3 + k]);
		}
		printf("\n");
	}

	return output;
}

objLoader * parseObj(){
	objLoader *objData = new objLoader();
	objData->load("test.obj");

	// 
	const int faceAmount = objData->faceCount;


	printf("Number of vertices: %i\n", objData->vertexCount);
	printf("Number of vertex normals: %i\n", objData->normalCount);
	printf("Number of texture coordinates: %i\n", objData->textureCount);
	printf("\n");
	
	printf("Number of faces: %i\n", objData->faceCount);

	return objData;
}

int main ()
{
	/* Initalize Platform IDs */ 
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}

	/* Initalize Device IDs */
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}


	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	CheckError (error);

	std::cout << "Context created" << std::endl;


	// Create a program from source
	cl_program program = CreateProgram (LoadKernel ("kernels/image.cl"),
		context);

	CheckError (clBuildProgram (program, deviceIdCount, deviceIds.data (), 
		"-D FILTER_SIZE=1", nullptr, nullptr));

	/* Send log to CMD window */
	size_t log_size;
	clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	char *log = (char *)malloc(log_size);

	clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	printf("%s\n", log);
	

	/* Create the Kernel */
	cl_kernel kernel = clCreateKernel (program, "Filter", &error);
	CheckError (error);

	// Image info
	static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
	static const int imageWidth = 512;
	static const int imageHeight = 512;

	cl_mem outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format,
		imageWidth, imageHeight, 0,
		nullptr, &error);
	CheckError(error);
	const auto image = RGBtoRGBA(LoadImage("test.ppm"));
	
	// 
	float SpherePosValue[] = { 0.4f, 1.0, 0.0, 0 };
	cl_mem SpherePos = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* 4, &SpherePosValue, &error);


	/* PARSING OBJECTS BITCHES */
	objLoader* loadedObject = parseObj();

	// export verts
	float* vertArray = VertsToFloat3(loadedObject->vertexList, loadedObject->vertexCount);

	// export faces
	int* faceArray = FacesToVerts(loadedObject->faceList, loadedObject->faceCount);

	// create buffers
	cl_mem faceData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*loadedObject->faceCount * 3, faceArray, &error);
	cl_mem vertData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*loadedObject->vertexCount * 3, vertArray, &error);
	cl_mem faceCount = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &loadedObject->faceCount, &error);

	// Free MALLOC when finished
	//free(vertArray);
	//free(faceArray);

	// Setup the kernel arguments
	clSetKernelArg (kernel, 0, sizeof (cl_mem), &outputImage);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &SpherePos);
	clSetKernelArg(kernel, 2, sizeof (cl_mem), &vertData);
	clSetKernelArg(kernel, 3, sizeof (cl_mem), &faceData);
	clSetKernelArg(kernel, 4, sizeof (cl_mem), &faceCount);
	
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateCommandQueue.html
	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
	CheckError (error);

	// Run the processing
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	std::size_t offset [3] = { 0 };
	std::size_t size [3] = { imageWidth, imageHeight, 1 };
	CheckError (clEnqueueNDRangeKernel (queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr));
	
	// Prepare the result image, set to black
	Image result = image;
	std::fill (result.pixel.begin (), result.pixel.end (), 0);

	// Get the result back to the host
	std::size_t origin [3] = { 0 };
	std::size_t region [3] = { result.width, result.height, 1 };
	clEnqueueReadImage (queue, outputImage, CL_TRUE,
		origin, region, 0, 0,
		result.pixel.data (), 0, nullptr, nullptr);

	// Save and finish up
	SaveImage (RGBAtoRGB (result), "output.ppm");

	std::cout << "Finished.  Press any key to continue" << std::endl;
	std::getchar();

	// Clean up
	clReleaseMemObject (outputImage);
	clReleaseCommandQueue (queue);
	
	clReleaseKernel (kernel);
	clReleaseProgram (program);
	clReleaseContext (context);
}