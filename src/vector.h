#pragma once
#include <cmath>

struct Vec3
{
    double x = 0, y = 0, z = 0;

    Vec3(double x, double y, double z);
    Vec3(const Vec3& v);

    // Vec3 @ Vec3
    Vec3 operator + (const Vec3& v) const;
    Vec3 operator * (const Vec3& v) const;
    Vec3 operator - (const Vec3& v) const;
    bool operator == (const Vec3& v) const;

    // Vec3 @ double
    Vec3 operator + (double d) const;
    Vec3 operator * (double d) const;
    Vec3 operator / (double d) const;

    double length() const;
    Vec3 normalize() const;
    Vec3 abs() const;
    Vec3 sqrt() const;

    //static double dot(const Vec3& a, const Vec3& b);

    static inline double dot(const Vec3& a, const Vec3& b)
    {
        return (a.x*b.x + a.y*b.y + a.z*b.z);
    };
};

Vec3 operator * (double d, const Vec3& v);
