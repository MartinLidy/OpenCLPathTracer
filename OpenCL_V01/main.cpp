#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "objLoader.h"
#include "obj_parser.h"
#include "GL/freeglut.h"

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

static cl_command_queue command_queue = NULL;
static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
int width = 512;
int height = 408;
int spp = 0;
float* pixels = NULL;

// OpenCL stuff
cl_command_queue queue = NULL;
cl_int error = 0;
cl_kernel kernel = NULL;
cl_context context = NULL;
cl_program program;

// CL MEMS
static cl_mem outputImage = NULL;
static cl_mem outputImage2 = NULL;
static cl_mem mem_rand_states = NULL; //random number states
static unsigned int* rand_states = NULL; //local
static cl_mem mem_image = NULL;

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

static size_t RoundUp(int groupSize, int globalSize) {
	int r = globalSize % groupSize;
	if (r == 0) { //no remainder
		return globalSize;
	}
	else {
		return globalSize + groupSize - r;
	}
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


float *VertsToFloat3(obj_vector** verts, int vertCount){
	int arraySize = vertCount * 3;
	float *output = (float*)malloc(arraySize * sizeof(float));

	// Each vert
	for (int i = 0; i < vertCount; i++){
		//printf("Vert: %d = [", i);

		// Each coord
		for (int k = 0; k < 3; k++){
			output[i * 3 + k] = verts[i]->e[k];
			//printf("%f, ", output[i * 3 + k]);
		}
		//printf("]\n");
	}

	return output;
}

int *FacesToVerts(obj_face** faces, int faceCount){
	int arraySize = faceCount * 3;
	int *output = (int*)malloc(arraySize * sizeof(int));

	// Each face
	for (int i = 0; i < faceCount; i++){
		//printf("Face: %d\n", i);

		// Each vert
		for (int k = 0; k < 3; k++){
			output[i * 3 + k] = faces[i]->vertex_index[k];
			//printf("   Vert: %d", output[i * 3 + k]);
		}
		//printf("\n");
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

	printf("Material List: %d\n", objData->materialList[objData->faceList[0]->material_index]->spec);

	return objData;
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

	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

static void SetupViewport(void) {
	glViewport(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
	glLoadIdentity(); //reset matrix
	//multiply current matrix by orthographic matrix, set clipping planes
	glOrtho(0.f, glutGet(GLUT_WINDOW_WIDTH) - 1.f, 0.f, glutGet(GLUT_WINDOW_HEIGHT) - 1.f, -1.f, 1.f);
}

static void AllocateLocalImageMem(void) {
	outputImage2 = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, width, height, 0, NULL, &error);
	Image result2 = RGBtoRGBA(LoadImage("test.ppm"));

	pixels = (float*)malloc(result2.width * result2.height * 4 * sizeof(float));
	int bytes = (result2.width * result2.height * 4 * sizeof(float));
	memset(pixels, 0, bytes); //set pixel colour

	CheckError(error);
	if (error != CL_SUCCESS) {
		printf("OpenCL: Error allocating image2d_t image memory\n");
	}
}

static void DrawImage(void) {
	printf("Drawing Image!");

	printf("%i", sizeof(pixels));
	
	Image result2 = RGBtoRGBA(LoadImage("test.ppm"));

	glColor3f(1.f, 1.f, 1.f);
	glEnable(GL_TEXTURE_2D); //enable tex. mapping
	glBindTexture(GL_TEXTURE_2D, 0);
	glPixelStoref(GL_UNPACK_ALIGNMENT, 4);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, result2.width, result2.height, 0, GL_RGBA, GL_FLOAT, pixels);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 0.0, 0.0);

	glTexCoord2f(1.0, 0.0);
	glVertex3f(glutGet(GLUT_WINDOW_WIDTH), 0.0, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 0.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(0.0, glutGet(GLUT_WINDOW_HEIGHT), 0.0);
	glEnd();
	glDisable(GL_TEXTURE_2D); //disable tex. mapping
}

static void Display_cb(void) {
	std::cout << "[Display Thread]" << std::endl;

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0); //set raster drawing position
	DrawImage();

	//swap buffers, for double buffering
	glutSwapBuffers();
}


//TODO figure out more efficient way of averaging samples
void UpdateLocalPixels(void) {
	cl_int error = 0;
	const size_t origin[3] = { 0, 0, 0 };
	const size_t region[3] = { width, height, 1 };
	float* old_pixels = (float*)malloc(width * height * 4 * sizeof(float));
	memcpy(old_pixels, pixels, width * height * 4 * sizeof(float));

	//local pixel array should already be allocated
	error = clEnqueueReadImage(queue, outputImage2, CL_TRUE, origin, region, 0, 0, pixels, 0, NULL, NULL);
	
	if (error != CL_SUCCESS) {
		printf("OpenCL: Error reading output from outputImage into local variable\n");
		exit(error);
	}
	if (spp > 0) {
		int i, j;
		for (i = 0; i < width*height; i++) {
			for (j = 0; j < 4; j++) {
				pixels[i * 4 + j] = (old_pixels[i * 4 + j] * (spp - 1) + pixels[i * 4 + j]) / (float)spp;
			}
		}
	}
	free(old_pixels);
	old_pixels = NULL;
	spp++;
}


int runKernelOriginal(){
	// Run the processing
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	std::size_t offset[3] = { 0 };
	std::size_t size[3] = { width, height, 1 };

	std::cout << "NDRANGE KERNEL STUFF " << std::endl;
	CheckError(clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr));

	std::cout << "NDRANGE KERNEL STUFF PASSED!" << std::endl;
	
	// Prepare the result image, set to black
	Image result = RGBtoRGBA(LoadImage("test.ppm"));
	std::fill(result.pixel.begin(), result.pixel.end(), 0);

	std::cout << "NDRANGE KERNEL STUFF BLAH " << std::endl;

	// Get the result back to the host
	std::size_t origin[3] = { 0, 0, 0 };
	std::size_t region[3] = { result.width, result.height, 1 };
	error = clEnqueueReadImage(queue, outputImage, CL_TRUE,origin, region, 0, 0, pixels, 0, nullptr, nullptr);
	CheckError(error);

	std::cout << "NDRANGE KERNEL STUFF BLAH BLAH" << std::endl;

	// Save and finish up
	SaveImage(RGBAtoRGB(result), "output.ppm");

	std::cout << "Finished.  Press any key to continue" << std::endl;
	std::getchar();

	// Clean up
	clReleaseMemObject(outputImage);
	clReleaseCommandQueue(queue);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}

int runKernel(){
	// Run the processing
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	std::size_t offset[3] = { 0 };
	std::size_t size[3] = { width, height, 1 };

	std::cout << "About to do stuff with Queque /n" << std::endl;
	error = clEnqueueNDRangeKernel(queue, kernel, 2, offset, size, nullptr, 0, nullptr, nullptr);
	CheckError(error);

	Image result = RGBtoRGBA(LoadImage("test.ppm"));
	
	// Prepare the result image, set to black
	//std::fill(result.pixel.begin(), result.pixel.end(), 0);

	// Get the result back to the host
	std::size_t origin[3] = { 0 };
	std::size_t region[3] = {result.width, result.height, 1 };

	// Get the result back to the host
	error = clEnqueueReadImage(queue, outputImage, CL_TRUE,origin, region, 0, 0,result.pixel.data(), 0, nullptr, nullptr);

	// Save and finish up
	SaveImage(RGBAtoRGB(result), "output.ppm");

	std::cout << "Finished.  Press any key to continue" << std::endl;
	std::getchar();

	std::cout << "About to do stuff with Queque again /n" << std::endl;

	// Save image into pixels
	//error = clEnqueueReadImage(queue, outputImage2, CL_TRUE, origin, region, 0, 0, pixels, 0, NULL, NULL);
	//error = clEnqueueReadImage(queue, outputImage, CL_TRUE, origin, region, 0, 0, pixels, 0, nullptr, nullptr);
	CheckError(error);
	std::cout << "Finished reading image" << std::endl;

	clFlush(queue); //issue all queued opencl commands to device
	clFinish(queue); //wait till processing is done

	//UpdateLocalPixels();

	//std::cout << "About to update local pixel array" << std::endl;
	
	/*clEnqueueReadImage(queue, outputImage2, CL_TRUE, origin, region, 0, 0, pixels, 0, nullptr, nullptr);

	clFlush(queue); //issue all queued opencl commands to device
	clFinish(queue); //wait till processing is done
	std::cout<<"About to update local pixel array"<<std::endl;

	UpdateLocalPixels(); //update local image*/

	return 0;
}



static void ExecuteKernel(void) {
	printf("OpenCL: Executing kernel\n");
	cl_int error = 0;

	size_t num_local_work_items[2] = { 16, 16 };
	size_t num_global_work_items[2] = { RoundUp(num_local_work_items[0], width),
		RoundUp(num_local_work_items[1], height) };

	error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		num_global_work_items, num_local_work_items, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("OpenCL: Error enqueuing kernel to command queue\n");
		exit(error);
	}
}

void OpenCLRender(void) {
	printf("OpenCL: Called OpenCLRender(void)\n");
	ExecuteKernel(); //execute kernel to command queue
	clFlush(queue); //issue all queued opencl commands to device
	clFinish(queue); //wait till processing is done
	//cout<<"About to update local pixel array"<<endl;

	UpdateLocalPixels(); //update local image
}

void OpenCLResetRender(void) {
	cl_int error = 0;
	spp = 0; //reset samples per pixel count

	//free memory, reallocate
	//--------------------------------------Image
	clReleaseMemObject(outputImage);
	outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, width, height, 0, NULL, &error);
}


static void UpdateKernel(void) {
	std::cout << "[Update Kernel]" << std::endl;
	width = glutGet(GLUT_WINDOW_WIDTH);
	height = glutGet(GLUT_WINDOW_HEIGHT);

	//OpenCLResetRender();
	//OpenCLRender(); //render pass
	runKernel();
	//runKernelOriginal();

	std::cout << "Background Thread" << std::endl;
	
	glutPostRedisplay(); //show new render on-screen
}


//Set callbacks
static void SetGLUTCallbacks(void) {
	std::cout << "Setup Callbacks" << std::endl;
	glutDisplayFunc(Display_cb);
	//glutReshapeFunc(Resize_cb);
	//glutKeyboardFunc(Keyboard_cb);
	//glutMotionFunc(Mouse_cb);
	//glutPassiveMotionFunc(Mouse_cb);
	glutIdleFunc(UpdateKernel);
}


static void SetupWindow(int argc, char **argv) {
	float width = 512;
	float height = 512;

	glutInit(&argc, argv); //parses window-system specific parameters
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(150, 50);
	glutCreateWindow("OPENCL Render V0.000000001");
}

void SetupGLUT(int argc, char** argv) {
	std::cout << "Setup Window" << std::endl;
	SetupWindow(argc, argv);
	
	std::cout << "Setup GLUT" << std::endl;
	SetGLUTCallbacks();

	SetupViewport();
	glutSetCursor(GLUT_CURSOR_NONE);
}


int setupOpenCL(){
	
	/* Initalize Platform IDs */
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	}
	else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
	}

	/* Initalize Device IDs */
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	}
	else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data(), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	context = clCreateContext(contextProperties, deviceIdCount,
		deviceIds.data(), nullptr, nullptr, &error);
	CheckError(error);

	std::cout << "Context created" << std::endl;


	// Create a program from source
	program = CreateProgram(LoadKernel("kernels/image.cl"),
		context);

	CheckError(clBuildProgram(program, deviceIdCount, deviceIds.data(),
		"-D FILTER_SIZE=1", nullptr, nullptr));

	std::cout << "Program created" << std::endl;

	/* Send log to CMD window */
	/*size_t log_size;
	clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	char *log = (char *)malloc(log_size);

	clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	printf("%s\n", log);*/

	/* Create the Kernel */
	kernel = clCreateKernel(program, "Filter", &error);
	CheckError(error);
	std::cout << "Kernel Created" << std::endl;

	// Image info
	std::cout << "Loading Image" << std::endl;
	Image image = RGBtoRGBA(LoadImage("test.ppm"));

	// Set image size
	height = image.height;
	width = image.width;

	outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, width, height, 0,pixels, &error);
	CheckError(error);

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
	cl_mem faceMatData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*loadedObject->faceCount * 3, faceArray, &error);

	// Setup the kernel arguments
	clSetKernelArg(kernel, 0, sizeof (cl_mem), &outputImage);
	clSetKernelArg(kernel, 1, sizeof (cl_mem), &vertData);
	clSetKernelArg(kernel, 2, sizeof (cl_mem), &faceData);
	clSetKernelArg(kernel, 3, sizeof (cl_mem), &faceCount);
	clSetKernelArg(kernel, 4, sizeof (cl_mem), &faceMatData);

	// Free MALLOC when finished
	free(vertArray);
	free(faceArray);

	std::cout << "Arguments Passed to Kernel" << std::endl;

	// Start the Queue
	queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);
	CheckError(error);

	return 0;
}

int main ()
{
	std::cout << "Starting OpenCL" << std::endl;
	
	// INIT Opencl
	setupOpenCL();
	AllocateLocalImageMem(); //allocate pixel array

	// Windowing system
	std::cout << "GLUT" << std::endl;
	SetupGLUT(0, nullptr);	

	// Create Kernel
	//createKernel();

	// Main Loop
	glutMainLoop();

	//std::cout << "Finished.  Press any key to continue" << std::endl;
	//std::getchar();

	// RUns Kernel
	//runKernel();
}
