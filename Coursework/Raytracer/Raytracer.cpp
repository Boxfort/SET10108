#include "stdafx.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <thread>

using namespace std;
using namespace std::chrono;

constexpr std::size_t MAX_DEPTH = 512; // Upper limit on recursion, increase this on systems with more stack size.
constexpr double PI = 3.14159265359;

template <class T, class Compare>
constexpr const T &clamp(const T &v, const T &lo, const T &hi, Compare comp)
{
	return assert(!comp(hi, lo)), comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template <class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi)
{
	return clamp(v, lo, hi, std::less<>());
}

struct vec
{
	__m128 components;

	//Working.
	vec(float x = 0.0f, float y = 0.0f, float z = 0.0f) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(float) * 4, 16);
		components = _mm_set_ps(0.0f, z, y, x);
		/*
		components.m128_f32[0] = x;
		components.m128_f32[1] = y;
		components.m128_f32[2] = z;
		components.m128_f32[3] = 0.0f;
		*/
	}

	//Working.
	vec(__m128 vector) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(float) * 4, 16);
		components = vector;
	}

	//vec(const vec& copy)
	//{
		//components = *(__m128*)_aligned_malloc(sizeof(float) * 4, 16);
	//	components = copy.components;
	//}

	float get_x() noexcept
	{
		return components.m128_f32[0];
	}

	float get_y() noexcept
	{
		return components.m128_f32[1];
	}

	float get_z() noexcept
	{
		return components.m128_f32[2];
	}

	//Working.
	vec operator+(const vec other) const noexcept
	{
		return vec(_mm_add_ps(other.components, components));
	}

	vec operator-(const vec other) const noexcept
	{
		return vec(_mm_sub_ps(other.components, components));
	}

	vec operator*(float scale) const noexcept
	{
		return vec(_mm_mul_ps(components, _mm_set1_ps(scale)));
	}

	vec mult(const vec &other) const noexcept
	{
		return vec(_mm_mul_ps(other.components, components));
	}

	vec normal() const noexcept
	{
		// Square each component
		__m128 result = _mm_mul_ps(components, components);
		// Sum all components
		result.m128_f32[0] = result.m128_f32[1] = result.m128_f32[2] = result.m128_f32[0] + result.m128_f32[1] + result.m128_f32[2];
		// Find reciprocal squrare root and times by original vector
		return vec(_mm_mul_ps(_mm_rsqrt_ps(result), components));
	}

	float dot(const vec &other) const noexcept
	{
		__m128 result = _mm_mul_ps(other.components, components);
		return result.m128_f32[0] + result.m128_f32[1] + result.m128_f32[2];
	}

	vec cross(const vec &other) const noexcept
	{
		return vec(components.m128_f32[1] * other.components.m128_f32[2] - components.m128_f32[2] * other.components.m128_f32[1], components.m128_f32[2] * other.components.m128_f32[0] - components.m128_f32[0] * other.components.m128_f32[2], components.m128_f32[0] * other.components.m128_f32[1] - components.m128_f32[1] * other.components.m128_f32[0]);
		//return vec(_mm_sub_ps(
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 1, 0, 2))),
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 0, 2, 1)))
		//));
	}

};

struct ray
{
	vec origin, direction;

	ray(const vec &origin, const vec &direction) noexcept
		: origin(origin), direction(direction)
	{
	}
};

enum struct reflection_type { DIFFUSE, SPECULAR, REFRACTIVE };

struct sphere
{
	double radius;
	vec position;
	vec emission, colour;
	reflection_type reflection;

	sphere(double radius, vec position, vec emission, vec colour, reflection_type reflection) noexcept
		: radius(radius), position(position), emission(emission), colour(colour), reflection(reflection)
	{
	}

	double intersection(const ray &ray) const noexcept
	{
		static constexpr double eps = 1e-4;
		vec origin_position = position - ray.origin;
		double b = origin_position.dot(ray.direction);
		double determinant = b * b - origin_position.dot(origin_position) + radius * radius;
		if (determinant < 0)
		{
			return 0;
		}
		else
		{
			determinant = sqrt(determinant);
		}
		double t = b - determinant;
		if (t > eps)
		{
			return t;
		}
		else
		{
			t = b + determinant;
			if (t > eps)
			{
				return t;
			}
			else
			{
				return 0;
			}
		}
	}
};

inline bool intersect(const vector<sphere> &spheres, const ray &ray, double &distance, std::size_t &sphere_index) noexcept
{
	static constexpr double maximum_distance = 1e20;
	distance = maximum_distance;
	for (std::size_t index = 0; index < spheres.size(); ++index)
	{
		double temp_distance = spheres[index].intersection(ray);
		if (temp_distance > 0 && temp_distance < distance)
		{
			distance = temp_distance;
			sphere_index = index;
		}
	}
	return distance < maximum_distance;
}

vec radiance(const vector<sphere> &spheres, const ray &the_ray, int depth) noexcept
{
	static random_device rd;
	static default_random_engine generator(rd());
	static uniform_real_distribution<double> distribution;
	static auto get_random_number = bind(distribution, generator);

	double distance;
	std::size_t sphere_index;
	if (!intersect(spheres, the_ray, distance, sphere_index))
		return vec();
	const sphere &hit_sphere = spheres[sphere_index];
	vec hit_point = the_ray.origin + the_ray.direction * distance;
	vec intersection_normal = (hit_point - hit_sphere.position).normal();
	vec pos_intersection_normal = intersection_normal.dot(the_ray.direction) < 0 ? intersection_normal : intersection_normal * -1;
	vec colour = hit_sphere.colour;
	double max_reflection = max({ colour.get_x(), colour.get_y(), colour.get_z() });

	if (sphere_index == 1)
	{
		cout << "" << endl;
	}

	if (depth > MAX_DEPTH)
	{
		return hit_sphere.emission;
	}
	else if (++depth > 5)
	{
		if (get_random_number() < max_reflection)
		{
			colour = colour * (1.0 / max_reflection);
		}
		else
		{
			return hit_sphere.emission;
		}
	}

	if (hit_sphere.reflection == reflection_type::DIFFUSE)
	{
		double r1 = 2.0 * PI * get_random_number();
		double r2 = get_random_number();
		vec w = pos_intersection_normal;
		vec u = ((abs(w.get_x()) > 0.1 ? vec(0, 1, 0) : vec(1, 0, 0)).cross(w)).normal();
		vec v = w.cross(u);
		vec new_direction = (u * cos(r1) * sqrt(r2) + v * sin(r1) * sqrt(r2) + w * sqrt(1 - r2)).normal();
		return hit_sphere.emission + colour.mult(radiance(spheres, ray(hit_point, new_direction), depth));
	}
	else if (hit_sphere.reflection == reflection_type::SPECULAR)
	{
		return hit_sphere.emission + colour.mult(radiance(spheres, ray(hit_point, the_ray.direction - intersection_normal * 2 * intersection_normal.dot(the_ray.direction)), depth));
	}
	ray reflection_ray(hit_point, the_ray.direction - intersection_normal * 2 * intersection_normal.dot(the_ray.direction));
	bool into = intersection_normal.dot(pos_intersection_normal) > 0;
	double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc;
	double ddn = the_ray.direction.dot(pos_intersection_normal);
	double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
	if (cos2t < 0.0)
	{
		return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth));
	}
	vec tdir = (the_ray.direction * nnt - intersection_normal * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).normal();
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1 - (into ? -ddn : tdir.dot(intersection_normal));
	double Re = R0 + (1 - R0) * c * c * c * c * c;
	double Tr = 1 - Re;
	double P = 0.25 + 0.5 * Re;
	double RP = Re / P;
	double TP = Tr / (1.0 - P);
	if (depth > 2)
	{
		if (get_random_number() < P)
		{
			return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth) * RP);
		}
		else
		{
			return hit_sphere.emission + colour.mult(radiance(spheres, ray(hit_point, tdir), depth) * TP);
		}
	}
	else
	{
		return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth) * Re + radiance(spheres, ray(hit_point, tdir), depth) * Tr);
	}
}

struct lwrite
{
	unsigned long value;
	unsigned size;

	lwrite(unsigned long value, unsigned size) noexcept
		: value(value), size(size)
	{
	}
};

inline std::ostream &operator<<(std::ostream &outs, const lwrite &v)
{
	unsigned long value = v.value;
	for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
		outs.put(static_cast<char>(value & 0xFF));
	return outs;
}

bool array2bmp(const std::string &filename, const vector<vec> &pixels, const std::size_t width, const std::size_t height)
{
	std::ofstream f(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	if (!f)
	{
		return false;
	}
	// Write Bmp file headers
	const std::size_t headers_size = 14 + 40;
	const std::size_t padding_size = (4 - ((height * 3) % 4)) % 4;
	const std::size_t pixel_data_size = width * ((height * 3) + padding_size);
	f.put('B').put('M'); // bfType
						 // bfSize
	f << lwrite(headers_size + pixel_data_size, 4);
	// bfReserved1, bfReserved2
	f << lwrite(0, 2) << lwrite(0, 2);
	// bfOffBits, biSize
	f << lwrite(headers_size, 4) << lwrite(40, 4);
	// biWidth,  biHeight,  biPlanes
	f << lwrite(width, 4) << lwrite(height, 4) << lwrite(1, 2);
	// biBitCount, biCompression = BI_RGB ,biSizeImage
	f << lwrite(24, 2) << lwrite(0, 4) << lwrite(pixel_data_size, 4);
	// biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant
	f << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4);
	// Write image data
	for (std::size_t x = height; x > 0; x--)
	{
		for (std::size_t y = 0; y < width; y++)
		{
			const auto &val = pixels[((x - 1) * width) + y];
			f.put(static_cast<char>(int(255.0f * val.components.m128_f32[2]))).put(static_cast<char>(int(255.0f * val.components.m128_f32[1]))).put(static_cast<char>(255.0f * val.components.m128_f32[0]));
		}
		if (padding_size)
		{
			f << lwrite(0, padding_size);
		}
	}
	return f.good();
}

int main(int argc, char **argv)
{
	/*
		--- TESTING ---
	*/

	vec a = vec(1.0, 2.0, 1.0);
	vec b = vec(2.0, 2.0, 1.0);

	vec c = a.mult(b);

	if (c.get_x() != 2.0f || c.get_y() != 4.0f || c.get_z() != 1.0f)
		throw new exception;

	float dot = a.dot(b);

	if (dot != 7.0f)
		throw new exception;

	vec cross = a.cross(b);

	if (cross.get_x() != 0 || cross.get_y() != 1.0f || cross.get_z() != -2.0f)
		throw new exception;

	float clampa = clamp(0.002f, 0.0f, 1.0f); //0.002
	float clampb = clamp(-10.0f, 0.0f, 1.0f); //0.0
	float clampc = clamp(12.2f, 0.0f, 1.0f); // 1.0

	if (clampa != 0.002f || clampb != 0.0f || clampc != 1.0f)
		throw new exception;

	/*
		-- END TESTING ---
	*/


	random_device rd;
	default_random_engine generator(rd());
	uniform_real_distribution<double> distribution;
	auto get_random_number = bind(distribution, generator);

	// *** These parameters can be manipulated in the algorithm to modify work undertaken ***
	constexpr std::size_t dimension = 1024;
	constexpr std::size_t samples = 2; // Algorithm performs 4 * samples per pixel.
	vector<sphere> spheres
	{
		// ******** !!!!!!!!! COLOR IS BEING SET TO 0,0,0 IN THE RADIANCE FUNCTION FIX IT !!!!!!!!!!!!************
		sphere(1e5, vec(1e5f + 1.0f, 40.8f, 81.6f), vec(0.0f, 0.0f, 0.0f), vec(0.75f, 0.25f, 0.25f), reflection_type::DIFFUSE),
		sphere(1e5, vec(-1e5f + 99.0f, 40.8f, 81.6f), vec(), vec(0.25f, 0.25f, 0.75f), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, 1e5), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, -1e5 + 170), vec(), vec(), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 1e5, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, -1e5 + 81.6, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(16.5, vec(27, 16.5, 47), vec(), vec(1, 1, 1) * 0.999, reflection_type::SPECULAR),
		sphere(16.5, vec(73, 16.5, 78), vec(), vec(1, 1, 1) * 0.999, reflection_type::REFRACTIVE),
		sphere(600, vec(50, 681.6 - 0.27, 81.6), vec(12, 12, 12), vec(), reflection_type::DIFFUSE)
	};
	// **************************************************************************************

	ray camera(vec(50.0f, 52.0f, 295.6f), vec(0.0f, -0.042612f, -1.0f).normal());
	vec cx = vec(0.5135f);
	vec cy = (cx.cross(camera.direction)).normal() * 0.5135;
	vec r = vec();
	vector<vec> pixels(dimension * dimension);

	unsigned int NUM_THREADS = thread::hardware_concurrency();
	auto start = system_clock::now();

	//For each row of pixels.
//#pragma omp parallel for num_threads(NUM_THREADS) shared(pixels) private(r)
	for (int y = 0; y < dimension; ++y)
	{
		std::cout << "Rendering " << dimension << " * " << dimension << "pixels. Samples:" << samples * 4 << " spp (" << 100.0 * y / (dimension - 1) << ")" << endl;
		// For each pixel in row
		for (int x = 0; x < dimension; ++x)
		{
			for (std::size_t sy = 0, i = (dimension - y - 1) * dimension + x; sy < 2; ++sy)
			{
				for (std::size_t sx = 0; sx < 2; ++sx)
				{
					r = vec();

					//Repeat for sample count.
					for (std::size_t s = 0; s < samples; ++s)
					{
						double r1 = 2 * get_random_number(); 
						double r2 = 2 * get_random_number(); 
						double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

						vec direction = cx * static_cast<double>(((sx + 0.5 + dx) / 2 + x) / dimension - 0.5) + cy * static_cast<double>(((sy + 0.5 + dy) / 2 + y) / dimension - 0.5) + camera.direction;
						
						ray rayman = ray(camera.origin + direction * 140, direction.normal());
						vec radiance1 = radiance(spheres, ray(camera.origin + direction * 140, direction.normal()), 0);
						vec radiance2 = radiance1 * (1.0 / samples);

						r = r + radiance(spheres, ray(camera.origin + direction * 140, direction.normal()), 0) * (1.0 / samples);

						if (r.get_x() != 0.0f)
							throw new exception;
					}

					pixels[i] = pixels[i] + vec(clamp(r.get_x(), 0.0f, 1.0f), clamp(r.get_y(), 0.0f, 1.0f), clamp(r.get_z(), 0.0f, 1.0f)) * 0.25f;

				}
			}
		}
	}

	auto end = system_clock::now();
	auto total = end - start;

	std::cout << "img.bmp" << (array2bmp("img.bmp", pixels, dimension, dimension) ? " Saved\n" : " Save Failed\n");
	//std::cout << "Time taken: " << duration_cast<seconds>(total).count() << endl;

	return 0;
}