#include "Vector4.h"
#include "Vector2.h"
#include "Vector3.h"

template<typename T>
Eule::Vector4<T>::operator Eule::Vector2<T>() const
{
	return Vector2<T>(x, y);
}

template<typename T>
Eule::Vector4<T>::operator Eule::Vector3<T>() const
{
	return Vector3<T>(x, y, z);
}
