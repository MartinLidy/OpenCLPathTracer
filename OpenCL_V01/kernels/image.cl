__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


float plane1(float3 planePos, float3 rayDir, float3 rayOrigin, float depth)
{
	float scale = 0.01;
	float t = dot(rayOrigin,(float3)(0.0,1.0,0.0)) / dot(rayDir,(float3)(0.0,1.0,0.0));
	t = -1*t;

	float3 hit = rayOrigin + t*rayDir;
	
	if(depth < 0.01){
		if(t < 0.00001f){
			return 0.1;
		}else{
			if (fmod(round(fabs(hit.x)*scale) + round(fabs(hit.z)*scale), 2.0f) < 1.0){
				return 0.0;
			}
			else{
				return 1.0 - (hit.x*0.001);
			}
		}

		return 0.1;
	}
	return 0;
}

float triangle(float3 v0, float3 v1, float3 v2, float3 ro, float3 rd)//, float3 *hit, float *dist)
{
	float3 position = (float3)(5000.0, 0.0, -500000.0);
	float output1 = 1.0;
	float scale = 0.1;

	v0 = position + v0*scale;
	v1 = position + v1*scale;
	v2 = position + v2*scale;

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;

    float3 pvec = cross(rd, edge2).xyz;
    float det = dot(edge1, pvec);
	
    if (det < 0.0) output1 = 0.0;
    
    float invDet = 1.0 / det;
    
	// ------------
    float3 tvec = ro - v0;
    float u = dot(tvec, pvec) * invDet;
    
	if (u < 0 || u > 1) output1=0.0;
    
	// ------------
    float3 qvec = cross(tvec, edge1);
    float v = dot(rd, qvec) * invDet;
    
    if (v < 0 || u + v > 1) output1=0.0;
	
	//
    float dist = dot(edge2, qvec) * invDet;
    float3 hit  = ro + rd * (dist);

	//printf("distance = %f\n", dist);
	//printf("v0 = %2.2v3hlf,  v1 = %2.2v3hlf,  V0-V1 = %2.2v3hlf\n", v0,v1,v2);
    return output1;
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


__kernel void Filter ( 
	__write_only image2d_t output, 
	__constant float4* example,
	__constant float verts[],
	__constant int faces[],
	__constant int* faceCount)
{
	const int2 iResolution = {512,512};
    const int2 pos = {get_global_id(0), get_global_id(1)};

	//Geometry
	float3 v1;
	float3 v2;
	float3 v3;
	
	// MSAA
	int i = 0;
	int k = 0;
	float rx = 0.0;
	float ry = 0.0;
	int samples = 1;
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
	float3 rayOrigin = ViewPlane+(float3)(rx*AA_amount,ry*AA_amount,0);

	//
	float3 hit;
	float dist;
	
	// Floor
	for(i=0;i<samples;i++){
		
		// Random Sample
		//rx = 0.5-rand( screenCoords.xy*(i) );
		//ry = 0.5-rand( screenCoords.xy*(i) );

		// Uniform Sample
		rx = samples/2 - i;
		ry = samples/2 - i;
		
		for(k=0; k<*faceCount; k++){
			v1 = (float3)( verts[faces[3*k]],	verts[faces[3*k]+1],	verts[faces[3*k]+2]   );
			v2 = (float3)( verts[faces[3*k+1]],	verts[faces[3*k+1]+1],	verts[faces[3*k+1]+2] );
			v3 = (float3)( verts[faces[3*k+2]],	verts[faces[3*k+2]+1],	verts[faces[3*k+2]+2] );

			sum += (float4)( triangle(v1, v2, v3, rayOrigin, rayDir));
		}

		//sum += (float4)( plane1( (float3)(0.0), rayDir,  rayOrigin, sum.x));
	}
	
	sum = sum/samples;

	// Sphere
	//sum = sphere( CamOrigin + (float3)(pos.s0,pos.s1,1.0f),rayDir, (float3)(0.0), 2.0f, sum );

    write_imagef (output, (int2)(pos.x, pos.y), sum);
}