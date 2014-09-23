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

float * parseObj(){
	objLoader *objData = new objLoader();
	objData->load("test.obj");

	// 
	const int vertAmount = objData->vertexCount;
	float output[12 * 3 * 3] = {};

	printf("Number of vertices: %i\n", objData->vertexCount);
	printf("Number of vertex normals: %i\n", objData->normalCount);
	printf("Number of texture coordinates: %i\n", objData->textureCount);
	printf("\n");

	printf("Number of faces: %i\n", objData->faceCount);
	for (int i = 0; i<objData->faceCount; i++)
	{
		obj_face *o = objData->faceList[i];
		printf(" face ");
		for (int j = 0; j<3; j++)
		{
			printVector(objData->vertexList[o->vertex_index[j]]);

			for (int k = 0; k < 3; k++){
				output[3*j + k] = objData->vertexList[o->vertex_index[j]]->e[k];
			}
		}

		printf("\n");
	}

	printf("\n");

	printf("Number of spheres: %i\n", objData->sphereCount);
	for (int i = 0; i<objData->sphereCount; i++)
	{
		obj_sphere *o = objData->sphereList[i];
		printf(" sphere ");
		printVector(objData->vertexList[o->pos_index]);
		printVector(objData->normalList[o->up_normal_index]);
		printVector(objData->normalList[o->equator_normal_index]);
		printf("\n");
	}

	printf("\n");

	printf("Number of planes: %i\n", objData->planeCount);
	for (int i = 0; i<objData->planeCount; i++)
	{
		obj_plane *o = objData->planeList[i];
		printf(" plane ");
		printVector(objData->vertexList[o->pos_index]);
		printVector(objData->normalList[o->normal_index]);
		printVector(objData->normalList[o->rotation_normal_index]);
		printf("\n");
	}

	printf("\n");

	printf("Number of point lights: %i\n", objData->lightPointCount);
	for (int i = 0; i<objData->lightPointCount; i++)
	{
		obj_light_point *o = objData->lightPointList[i];
		printf(" plight ");
		printVector(objData->vertexList[o->pos_index]);
		printf("\n");
	}

	printf("\n");

	printf("Number of disc lights: %i\n", objData->lightDiscCount);
	for (int i = 0; i<objData->lightDiscCount; i++)
	{
		obj_light_disc *o = objData->lightDiscList[i];
		printf(" dlight ");
		printVector(objData->vertexList[o->pos_index]);
		printVector(objData->normalList[o->normal_index]);
		printf("\n");
	}

	printf("\n");

	printf("Number of quad lights: %i\n", objData->lightQuadCount);
	for (int i = 0; i<objData->lightQuadCount; i++)
	{
		obj_light_quad *o = objData->lightQuadList[i];
		printf(" qlight ");
		printVector(objData->vertexList[o->vertex_index[0]]);
		printVector(objData->vertexList[o->vertex_index[1]]);
		printVector(objData->vertexList[o->vertex_index[2]]);
		printVector(objData->vertexList[o->vertex_index[3]]);
		printf("\n");
	}

	printf("\n");

	if (objData->camera != NULL)
	{
		printf("Found a camera\n");
		printf(" position: ");
		printVector(objData->vertexList[objData->camera->camera_pos_index]);
		printf("\n looking at: ");
		printVector(objData->vertexList[objData->camera->camera_look_point_index]);
		printf("\n up normal: ");
		printVector(objData->normalList[objData->camera->camera_up_norm_index]);
		printf("\n");
	}

	printf("\n");

	printf("Number of materials: %i\n", objData->materialCount);
	for (int i = 0; i<objData->materialCount; i++)
	{
		obj_material *mtl = objData->materialList[i];
		printf(" name: %s", mtl->name);
		printf(" amb: %.2f ", mtl->amb[0]);
		printf("%.2f ", mtl->amb[1]);
		printf("%.2f\n", mtl->amb[2]);

		printf(" diff: %.2f ", mtl->diff[0]);
		printf("%.2f ", mtl->diff[1]);
		printf("%.2f\n", mtl->diff[2]);

		printf(" spec: %.2f ", mtl->spec[0]);
		printf("%.2f ", mtl->spec[1]);
		printf("%.2f\n", mtl->spec[2]);

		printf(" reflect: %.2f\n", mtl->reflect);
		printf(" trans: %.2f\n", mtl->trans);
		printf(" glossy: %i\n", mtl->glossy);
		printf(" shiny: %i\n", mtl->shiny);
		printf(" refact: %.2f\n", mtl->refract_index);

		printf(" texture: %s\n", mtl->texture_filename);
		printf("\n");
	}

	printf("\n");

	//vertex, normal, and texture test
	if (objData->textureCount > 2 && objData->normalCount > 2 && objData->faceCount > 2)
	{
		printf("Detailed face data:\n");

		for (int i = 0; i<3; i++)
		{
			obj_face *o = objData->faceList[i];
			printf(" face ");
			for (int j = 0; j<3; j++)
			{
				printf("%i/", o->vertex_index[j]);
				printf("%i/", o->texture_index[j]);
				printf("%i ", o->normal_index[j]);
			}
			printf("\n");
		}
	}

	return output;
}

int main ()
{
	parseObj();

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

	// obj parser
	//float objectDataValue[] = parseObj();
	//cl_mem objectData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* 12, &objectDataValue, &error);

	// Setup the kernel arguments
	clSetKernelArg (kernel, 0, sizeof (cl_mem), &outputImage);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &SpherePos);
	//clSetKernelArg(kernel, 2, sizeof (cl_mem), &objectData);
	
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