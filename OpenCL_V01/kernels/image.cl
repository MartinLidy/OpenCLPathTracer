__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


float plane1(float3 planePos, float3 rayDir, float3 rayOrigin)
{
	float scale = 0.01;
	float t = dot(rayOrigin,(float3)(0.0,1.0,0.0)) / dot(rayDir,(float3)(0.0,1.0,0.0));
	t = -1*t;

	float3 hit = rayOrigin + t*rayDir;
	
	if(t < 0.00001f){
		return 0.1;
	}else{
		if (fmod(round(fabs(hit.x)*scale) + round(fabs(hit.z)*scale), 2.0f) < 1.0){
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


/*float rand(float2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}*/


__kernel void Filter ( __write_only image2d_t output, __constant float4* example)
{
	const int2 iResolution = {512,512};
    const int2 pos = {get_global_id(0), get_global_id(1)};
	
	// MSAA
	int i = 0;
	float rx = 0.0;
	float ry = 0.0;
	int samples = 64;
	float AA_amount = 0.05;

	// Screen info
	float scx = (float)( pos.x / iResolution.x )*2.0 - 1.0;
	float scy = (float)( pos.y / iResolution.y )*2.0 - 1.0;

	float3 screenCoords = {scx,scy,0};
    float4 sum = (float4)(0.0f);

	//
	float3 CamOrigin = (float3)(500.0, -500.0, -500.0);
	float3 ViewPlane = CamOrigin + (float3)(-256,-256,-256.0);

	float3 rayDir = (ViewPlane+(float3)(pos.x,pos.y,0)) - CamOrigin;
	
	// Floor
	for(i=0;i<samples;i++){
		
		// Random Sample
		//rx = 0.5-rand( screenCoords.xy*(i) );
		//ry = 0.5-rand( screenCoords.xy*(i) );

		// Uniform Sample
		rx = samples/2 - i;
		ry = samples/2 - i;
		sum += (float4)( plane1( (float3)(0.0), rayDir, ViewPlane+(float3)(rx*AA_amount,ry*AA_amount,0) ));
	}
	
	sum = sum/samples;

	// Sphere
	//sum = sphere( CamOrigin + (float3)(pos.s0,pos.s1,1.0f),rayDir, (float3)(0.0), 2.0f, sum );

    write_imagef (output, (int2)(pos.x, pos.y), sum);
}