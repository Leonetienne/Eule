#include "Catch2.h"
#include <Eule/Math.h>

using namespace Eule;

/** Equivalence classes:
*		-- value > 0
*		-- value < 0
*		-- value == 0
*/

// Checks with a positive input
TEST_CASE("Positive_Value", "[Math][Abs]")
{
    REQUIRE(Math::Abs(45.0) == 45.0);
    return;
}

// Checks with a negative input
TEST_CASE("Negative_Value", "[Math][Abs]")
{
    REQUIRE(Math::Abs(-45.0) == 45);
    return;
}

// Checks with a zero input
TEST_CASE("Zero_Value", "[Math][Abs]")
{
    REQUIRE(Math::Abs(0.0) == 0.0);
    return;
}
