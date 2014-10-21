__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


// Floor plane
bool plane(float3 pos, float3 norm, float3 ro, float3 rd, float3 *hit, float *dist)
{
	*dist = dot(pos-ro, norm) / dot(rd, norm);
	
	if(*dist < 0.0000000001){ 
		return false;
	}
	*hit = ro + *dist * rd;
	return true;
}

// Geometry
bool triangle(float3 v0, float3 v1, float3 v2, float3 ro, float3 rd, float3 *hit, float *dist, float3 *norm)
{
	float3 cubePos = (float3)(0.0,-3.0,7.0);
	v0 = cubePos + v0;
	v1 = cubePos + v1;
	v2 = cubePos + v2;

	float3 edge1 = v2 - v0;
	float3 edge2 = v1 - v0;

	float3 pvec = cross(rd, edge2);

	float det = dot(edge1, pvec);

	if (det == 0) return false;

	float invDet = 1 / det;
	float3 tvec = ro - v0;

	float u = dot(tvec, pvec) * invDet;

	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec, edge1);
	float v = dot(rd, qvec) * invDet;

	if (v < 0 || u + v > 1) return false;

	*dist = dot(edge2, qvec) * invDet;
	*hit = ro + rd* (*dist);
	*norm = (float3)( edge1.y*edge2.z - edge1.z*edge2.y, edge1.z*edge2.x - edge1.x*edge2.z, edge1.x*edge2.y - edge1.y*edge2.x);

	return true;
}

// Implicit Sphere
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
	float3 N = -1.0f * normalize((float3)((hit.x - center.x)/radius, (hit.y - center.y)/radius, (hit.z - center.z)/radius));

	// Return lit sphere
	return (float4)(1.0,1.0,0.0,1.0);//(float4)(dot(N,L)* clamp(t,0.0f,1.0f)*att);
}

// Reflections
float3 reflectionFace(float3 rayDir, float3 N){
	float V = 0; // ?? Another missing variable
	float c1 = -dot( N, rayDir );
	float3 Rl = (float3)( V + (2 * N * c1 ) );
	return Rl;
}

// Refractions
float3 refractionFace(float3 rayDir, float3 N){
	float n1 = 0.1;//index of refraction of original medium
	float n2 = 0.5;//index of refraction of new medium
	float n = n1 / n2;
	float c1 = 0; // ??? WHATS THIS
	float c2 = sqrt( 1 - n2 * (1 - c1) );

	float3 Rr = (float3)( (n * rayDir) + (n * c1 - c2) * N );
	return Rr;
}

// Point lighting
float lightFace(float3 N, float3 Pos){

	// Light attenuation //
	float3 LPos = (float3)(0.0,0.0,7.0);
	float Ldist = length(LPos - Pos);
	float a = 0.1;
	float b = 0.01;
	float att = 1.0 / (1.0 + a*Ldist + b*Ldist*Ldist);
	
	return dot(N, LPos - Pos) * att;
}

// Find intersecting face
int getIntersection(float3 rayOrigin, float3 rayDir, float3* hit2, float3* norm2, __constant int* faces, __constant float* verts, __constant int* faceCount){//, *hit, *dist, *norm){
	float3 v1,v2,v3;
	float3 minHit, minNorm;
	float minDist = 999999.0;
	int k;

	float3 hit;
	float dist;
	float3 norm;

	int hitFaceIndex = -1;

	// For each face in faces array
	for(k=0; k<*faceCount; k++){
		v1 = (float3)( verts[3*faces[3*k]+0],	verts[3*faces[3*k]+1],		verts[3*faces[3*k]+2] );
		v2 = (float3)( verts[3*faces[3*k+1]+0],	verts[3*faces[3*k+1]+1],	verts[3*faces[3*k+1]+2] );
		v3 = (float3)( verts[3*faces[3*k+2]+0],	verts[3*faces[3*k+2]+1],	verts[3*faces[3*k+2]+2] );

		// Colision check
		if(triangle(v1, v2, v3, rayOrigin, rayDir, &hit, &dist, &norm)){
			if(dist < minDist){
				minDist = dist;
				minHit = hit;
				minNorm = norm;
				hitFaceIndex = k;
			}
		}
	}

	*hit2 = minHit;
	*norm2 = minNorm;

	return hitFaceIndex;
}

float3 getPointColor( int objIndex, __constant int* faceMat, __constant float* Materials ){
	
	// Floor
	if (objIndex == -1){
		return (float3)(1.0,0.9,0.9);
	}

	return (float3)( Materials[faceMat[objIndex]+0], Materials[faceMat[objIndex]+1], Materials[faceMat[objIndex]+2] );
}

float3 traceRay( float3 rayPos, float3 rayDir, __constant int* faces, __constant float* verts, __constant int* faceCount, __constant int* faceMat, __constant float* Materials )
{
	float3 reflect_color = (float3)(0.0);
	float3 refract_color = (float3)(0.0);
	float3 point_color = (float3)(0.0);
	float3 hit, norm, minHit, minNorm;
	float3 sum = (float3)(0.0);
	float dist;

	int objIndex;
	bool hitCube = false;

	objIndex = getIntersection( rayPos, rayDir, &hit, &norm, faces, verts, faceCount);

	// Didnt hit geometry
	if(objIndex != -1){
		point_color = getPointColor( objIndex, faceMat, Materials);
		point_color = point_color * lightFace(norm, hit);
	}

	// Hit the floor
	else if(plane( (float3)(0.0), normalize((float3)(0.0,0.0,1.0)), rayPos, rayDir, &hit, &dist) )
	{
		// Plane stuff
		float scale = 0.1;

		//do this calculation for all x, y, z, and it will work regardless of normal
		if ( fmod( round( fabs(hit.x)*scale) + round(fabs(hit.y)*scale) + round(fabs(hit.z)*scale), 2.0f) < 1.0){
			point_color = lightFace((float3)(0.0,0.0,1.0), hit);
		}	
		else{
			point_color = (float3)(1.0,0.0,0.0) * lightFace((float3)(0.0,0.0,1.0), hit);
		}
	}

	/*if ( object is reflective )
		reflect_color = trace_ray( get_reflected_ray( original_ray, obj ) )
	if ( object is refractive )
		refract_color = trace_ray( get_refracted_ray( original_ray, obj ) )*/

	//return ( combine_colors( point_color, reflect_color, refract_color ));

	//if(hitCube){
	//sum = (float4)(0.0,0.1,0.7,1.0) * lightFace(minNorm, minHit);

	return point_color;
}

__kernel void Filter ( 
	__write_only image2d_t output,
	__constant float verts[],
	__constant int faces[],
	__constant int* faceCount,
	__constant int* faceMat,
	__constant float Materials[])
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
	float3 camPos = (float3)(-2.0,-8.0,8.0);
	float3 forward = normalize((float3)(0.3,1.0,0.0));
	float3 up      = normalize((float3)(0.0,0.0,1.0));

	float3 right = normalize(cross(forward, up));
	up = normalize(cross(right, forward));
	float3 rayOrigin = camPos + forward;
	float3 rayDir = normalize(scx*right + scy*up + forward * (float3)(0.95));
	
	// Geometry //
	float4 sum = (float4)(0.0f);

	//
	float dist;
	float minDist = 10000000;
	bool hitCube = false;
	
	//This wasn't actually doing multisampling yet, so I commented it for now
	//for(i=0;i<samples;i++){
		
		// Random Sample  // Uniform Sample
		//rx = 0.5-rand( screenCoords.xy*(i) ); //rx = samples/2 - i;
		//ry = 0.5-rand( screenCoords.xy*(i) ); //ry = samples/2 - i;
		
		// Tracing
		sum.xyz = traceRay(rayOrigin, rayDir, faces, verts, faceCount, faceMat, Materials);
	//}
	
	//sum = sum/samples;

	// Lit Sphere
	//sum = sphere( camPos, rayDir, (float3)(0.0,-75.0,0.0), 0.1f);
	
    write_imagef (output, (int2)(pos.x, pos.y), sum);
	//write_imagef (output, (int2)(pos.x, pos.y), (float4)(1.0,0,0,1.0));
}