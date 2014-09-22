__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

float FilterValue (__constant const float* filterWeights,
	const int x, const int y)
{
	return filterWeights[(x+FILTER_SIZE) + (y+FILTER_SIZE)*(FILTER_SIZE*2 + 1)];
}


float plane1(float3 planePos, float3 rayDir, float3 rayOrigin)
{
	float scale = 0.01;
	float t = dot(rayOrigin,(float3)(0.0,1.0,0.0)) / dot(rayDir,(float3)(0.0,1.0,0.0));
	t = -1*t;

	float3 hit = rayOrigin + t*rayDir;
	if(t < 0.00001f){
		return 0.1;
	}else{
		if (fmod(round(hit.x*scale) + round(hit.z*scale), 2.0f) < 1.0){
			return 0.0;
		}
		else{
			return 0.3;
		}
	}

	return 0.1;
}


float4 sphere(float3 ray, float3 dir, float3 center, float radius, float4 previous)
{
	float3 rc = ray-center;
	float c = dot(rc, rc) - (radius*radius);
	float b = dot(dir, rc);
	float d = b*b - c;
	float t = -b - sqrt(fabs(d));
	float2 st = step(0.0, (float2)(fmin(t,d)));
	float4 color = (float4)(0);
	
	if(d>previous.s3){
		color = (float4)(clamp(d*0.0001,0.0,1.0));
		color.s3 = previous.s3;
	}
	else{
		color = previous;
	}
	return color;
}


__kernel void Filter (
	__read_only image2d_t input,
	__constant float* filterWeights,
	__write_only image2d_t output,
	__constant float4* example)
{
	const int2 iResolution = {512,408};
    const int2 pos = {get_global_id(0), get_global_id(1)};
	
	float scx = (float)( pos.x / iResolution.x )*2.0 - 1.0;
	float scy = (float)( pos.y / iResolution.y )*2.0 - 1.0;
	
	float3 screenCoords = {scx,scy,0};

	float4 sum2 = *example;
    float4 sum = (float4)(0.0f);

	//
	float3 CamOrigin = (float3)(0.0,	-150.0,	-150.0);
	float3 ViewPlane = CamOrigin + (float3)(-0.5,-0.5, 1.0);

	float3 rayDir = (ViewPlane) - CamOrigin;
	
	// Floor
	sum = (float4)( plane1( (float3)(0.0), rayDir, CamOrigin+(float3)(pos.x,pos.y,1.0f)));

	// Sphere
	//sum = sphere( CamOrigin + (float3)(pos.s0,pos.s1,1.0f),rayDir, (float3)(0.0), 2.0f, sum );

    write_imagef (output, (int2)(pos.x, pos.y), sum);
}