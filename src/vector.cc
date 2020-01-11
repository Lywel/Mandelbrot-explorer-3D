#include "vector.h"


//__device__ __host__
Vec3::Vec3(double x, double y, double z)
    : x(x), y(y), z(z)
{
}

//__device__ __host__
Vec3::Vec3(const Vec3& v)
    : x(v.x), y(v.y), z(v.z)
{
}

//__device__ __host__
Vec3
Vec3::operator + (const Vec3& v) const
{
    return Vec3(x+v.x, y+v.y, z+v.z);
}
Vec3
Vec3::operator * (const Vec3& v) const
{
    return Vec3(x*v.x, y*v.y, z*v.z);
}

//__device__ __host__
Vec3
Vec3::operator - (const Vec3& v) const
{
    return Vec3(x-v.x, y-v.y, z-v.z);
}
//__device__ __host__
bool
Vec3::operator == (const Vec3& v) const
{
    return (x==v.x && y==v.y && z==v.z);
}

//__device__ __host__
Vec3
Vec3::operator + (double d) const
{
    return Vec3(x+d, y+d, z+d);
}

//__device__ __host__
Vec3
Vec3::operator * (double d) const
{
    return Vec3(x*d, y*d, z*d);
}

//__device__ __host__
Vec3
Vec3::operator / (double d) const
{
    return Vec3(x/d, y/d, z/d);
}

//__device__ __host__
double
Vec3::length() const
{
    return std::sqrt(x*x + y*y + z*z);
}

//__device__ __host__
Vec3
Vec3::normalize() const
{
    double mg = std::sqrt(x*x + y*y + z*z);
    return Vec3(x/mg,y/mg,z/mg);
}

//__device__ __host__
Vec3
Vec3::abs() const
{
    return Vec3(x < 0 ? -x : x, y < 0 ? -y : y, z < 0 ? -z : z);
}

Vec3
Vec3::sqrt() const
{
    return Vec3(std::sqrt(x), std::sqrt(y), std::sqrt(z));
}

// Comutativity
Vec3 operator * (double d, const Vec3& v)
{
    return v * d;
}

