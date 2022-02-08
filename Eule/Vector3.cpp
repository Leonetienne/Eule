#include "Vector3.h"



#include "Vector3.h"
#include "Vector2.h"
#include "Vector4.h"

template<typename T>
Eule::Vector3<T>::operator Eule::Vector2<T>() const
{
	return Vector2<T>(x, y);
}

template<typename T>
Eule::Vector3<T>::operator Eule::Vector4<T>() const
{
	return Vector4<T>(x, y, z, 0);
}
