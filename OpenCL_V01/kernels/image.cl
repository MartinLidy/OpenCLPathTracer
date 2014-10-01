__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


bool plane(float3 pos, float3 norm, float3 ro, float3 rd, float3 *hit, float *dist)
{
	*dist = dot(pos-ro, norm) / dot(rd, norm);
	
	if(*dist < 0.0000000001){ 
		return false;
	}
	*hit = ro + *dist * rd;
	return true;
}

bool triangle(float3 v0, float3 v1, float3 v2, float3 ro, float3 rd, float3 *hit, float *dist)
{
	float scale = 0.10;
	v0 = v0*scale;
	v1 = v1*scale;
	v2 = v2*scale;

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 pvec = cross(rd, edge2).xyz;
    float det = dot(edge1, pvec);
    if (det < 0.0) return false;
    float invDet = 1.0 / det;
    float3 tvec = ro - v0;
    float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;
    float3 qvec = cross(tvec, edge1);
    float v = dot(rd, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;
	
	*dist = dot(edge2, qvec) * invDet;
    *hit  = ro + rd * (*dist);
    return true;
}

float4 sphere(float3 ray, float3 dir, float3 center, float radius)
{
	// parallel light direction
	float3 L = normalize((float3)(-1.0,-0.1,-0.2));

	//
	float3 rc = center;
	float c = dot(rc, rc) - (radius*radius);
	float b = dot(dir, rc);
	float d = b*b - c;
	float t = -b - sqrt(fabs(d));
	float st = step(0.0f, fmin(t,d));
	float4 color = (float4)(0);

	float3 hit = ray + dir*(float3)(t);

	//printf("Det=%f, Distance=%f\n",d,t);
	float3 N = -1.0f * normalize((float3)((hit.x - center.x)/radius, (hit.y - center.y)/radius, (hit.z - center.z)/radius));

	// Point light
	float3 LPos = (float3)(10,0,40);
	float dist = length(LPos - hit);
	float a2 = 0.1;
	float b2 = 0.1;
	float att = 1.0 / (1.0 + a2*dist + b2*dist*dist);
	//L = normalize(LPos - hit); // Point lighting

	// Return lit sphere
	return (float4)(dot(N,L)* clamp(t,0.0f,1.0f));
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
	// MSAA //
	int i = 0;
	int k = 0;
	float rx = 0.0;
	float ry = 0.0;
	int samples = 1;
	float AA_amount = 0.05;

	// Screen info //
	const int2 iResolution = {512,512};
    const int2 pos = {get_global_id(0), get_global_id(1)};
	float scx = ( (float)pos.x / iResolution.x )*2.0 - 1.0;
	float scy = ( (float)pos.y / iResolution.y )*-2.0 + 1.0;
	float3 screenCoords = {scx,scy,0};

	// Camera //
	float3 camPos = (float3)(0.0,-50.0,1.0);
	float3 forward = normalize((float3)(0.0,1.0,0.0));
	float3 up      = normalize((float3)(0.0,0.0,1.0));

	float3 right = normalize(cross(forward, up));
	up = normalize(cross(right, forward));
	float3 rayOrigin = camPos + forward;
	float3 rayDir = normalize(scx*right + scy*up + forward * (float3)(0.95));
	
	// Geometry //
	float4 sum = (float4)(0.0f);
	float3 v1;
	float3 v2;
	float3 v3;
	float3 hit;
	float dist;
	
	//This wasn't actually doing multisampling yet, so I commented it for now
	//for(i=0;i<samples;i++){
		
		// Random Sample  // Uniform Sample
		//rx = 0.5-rand( screenCoords.xy*(i) ); //rx = samples/2 - i;
		//ry = 0.5-rand( screenCoords.xy*(i) ); //ry = samples/2 - i;

		bool hitCube = false;
		float minDist = 10000000;
		float3 minHit = (float3)(0.0, 0.0, 0.0);

		// For each face in faces array
		for(k=0; k<*faceCount; k++){
			v1 = (float3)( verts[faces[3*k]],	verts[faces[3*k]+1],	verts[faces[3*k]+2]   );
			v2 = (float3)( verts[faces[3*k+1]],	verts[faces[3*k+1]+1],	verts[faces[3*k+1]+2] );
			v3 = (float3)( verts[faces[3*k+2]],	verts[faces[3*k+2]+1],	verts[faces[3*k+2]+2] );
			
			if(triangle(v1, v2, v3, rayOrigin, rayDir, &hit, &dist)){
				if(dist < minDist){
					minDist = dist;
					minHit = hit;
				}
				hitCube = true;
			}
		}

		if(false){//hitCube){
			sum = (float4)(0.0,0.1,0.7,1.0);
		}
		else if(plane( (float3)(0.0), normalize((float3)(0.0,0.0,1.0)), rayOrigin, rayDir, &hit, &dist) )
		{

			// Light attenuation
			float dist = length(hit);
			float a = 0.1;
			float b = 0.1;
			float att = 1.0 / (1.0 + a*dist + b*dist*dist);
			float3 LPos = (float3)(10,0,40);

			// Plane stuff
			float scale = 0.1;

			//do this calculation for all x, y, z, and it will work regardless of normal
			if ( fmod( round( fabs(hit.x)*scale) + round(fabs(hit.y)*scale) + round(fabs(hit.z)*scale), 2.0f) < 1.0){
				sum = dot((float3)(0.0,0.0,1.0), LPos - hit) * att;
			}	
			else{
				sum = (float4)(1.0,0.0,0.0,1.0)* dot((float3)(0.0,3.0,1.0), LPos - hit) * att;
			}
				 
		}
	//}
	
	//sum = sum/samples;

	// Lit Sphere
	//sum = sphere( camPos, rayDir, (float3)(0.0,100.0,1.0), 0.1f);
	
    write_imagef (output, (int2)(pos.x, pos.y), sum);
}