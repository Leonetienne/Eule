#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"

template<typename T>
Eule::Vector2<T>::operator Eule::Vector3<T>() const
{
	return Vector3<T>(x, y, 0);
}

template<typename T>
Eule::Vector2<T>::operator Eule::Vector4<T>() const
{
	return Vector4<T>(x, y, 0, 0);
}
