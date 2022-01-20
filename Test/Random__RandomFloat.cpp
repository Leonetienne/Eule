#include "CppUnitTest.h"
#include "../Eule/Random.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eule;

namespace _Random
{
	TEST_CLASS(_RandomFloat)
	{
	public:
		// Checks that all values are always 0 <= v <= 1
		TEST_METHOD(Always_Satisfies_0_lt_v_lt_1)
		{
			// Test 1000 random values
			for (std::size_t i = 0; i < 1e3; i++)
			{
				const double rnd = Random::RandomFloat();
				Assert::IsTrue(rnd >= 0.0, L"rnd < 0");
				Assert::IsTrue(rnd <= 1.0, L"rnd > 1");
			}

			return;
		}
	};
}