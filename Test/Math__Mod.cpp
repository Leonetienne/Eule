#include "CppUnitTest.h"
#include "../Eule/Math.h"
#include <stdexcept>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Eule;

/** Equivalence classes:
* a -> numerator
* b -> denominator
*		-- a > 0 && b > 0
*		-- a < 0 && b > 0
*		-- a > 0 && b < 0
*		-- a < 0 && b < 0
* 
*		-- a > 0 && b = 0
*		-- a = 0 && b > 0
* *		-- a < 0 && b = 0
*		-- a = 0 && b < 0
*		-- a = 0 && b = 0
*/

namespace _Math
{
	TEST_CLASS(_Mod)
	{
	public:
		// a > 0 && b > 0
		TEST_METHOD(a_gt_0_and_b_gt_0)
		{
			Assert::AreEqual(7, Math::Mod(199, 32));
			return;
		}

		// a < 0 && b > 0
		TEST_METHOD(a_lt_0_and_b_gt_0)
		{
			Assert::AreEqual(25, Math::Mod(-199, 32));
			return;
		}

		// a > 0 && b < 0
		TEST_METHOD(a_gt_0_and_b_lt_0)
		{
			Assert::AreEqual(-25, Math::Mod(199, -32));
			return;
		}

		// a > 0 && b = 0
		TEST_METHOD(a_gt_0_and_b_eq_0)
		{
			// Exppect divide-by-zero
			Assert::ExpectException<std::logic_error&>([]() {
				Assert::AreEqual(0, Math::Mod(199, 0));
			});
			return;
		}

		// a = 0 && b > 0
		TEST_METHOD(a_eq_0_and_b_gt_0)
		{
			Assert::AreEqual(0, Math::Mod(0, 32));
			return;
		}

		// a < 0 && b = 0
		TEST_METHOD(a_lt_0_and_b_eq_0)
		{
			// Exppect divide-by-zero
			Assert::ExpectException<std::logic_error&>([]() {
				Assert::AreEqual(0, Math::Mod(-199, 0));
				});
			return;
		}

		// a = 0 && b < 0
		TEST_METHOD(a_eq_0_and_b_lt_0)
		{
			Assert::AreEqual(0, Math::Mod(0, -32));
			return;
		}
	};
}
