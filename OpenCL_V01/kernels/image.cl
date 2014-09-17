__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

float FilterValue (__constant const float* filterWeights,
	const int x, const int y)
{
	return filterWeights[(x+FILTER_SIZE) + (y+FILTER_SIZE)*(FILTER_SIZE*2 + 1)];
}


float sphere(float3 ray, float3 dir, float3 center, float radius)
{
	float3 rc = ray-center;
	float c = dot(rc, rc) - (radius*radius);
	float b = dot(dir, rc);
	float d = b*b - c;
	float t = -b - sqrt(fabs(d));
	float2 st = step(0.0, (float2)(fmin(t,d)));
	
	
	//return mix(-1.0, t, st.s0);

	return (t + 1.0) * st.s0;

	//return 0.5f;
}


__kernel void Filter (
	__read_only image2d_t input,
	__constant float* filterWeights,
	__write_only image2d_t output,
	__constant float4* example)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 sum2 = *example;
    float4 sum = (float4)(0.0f);

	sum = (float4)( sphere(		(float3)(pos.s0,pos.s1,1.0f), (float3)(pos.s0,pos.s1,1.0f), (float3)(0.0), 100.0f )	);

    write_imagef (output, (int2)(pos.x, pos.y), sum);
}