#define BOOST_TEST_MODULE tfcpp

#include <iostream>

#include <boost/test/included/unit_test.hpp>

#include <objdet.hpp>

BOOST_AUTO_TEST_SUITE(tfcpp)

BOOST_AUTO_TEST_CASE(look_at_me_i_can_write_unittests)
{
    const string model_path = "../model";
    const string img_path = "../doggos.jpg";

    Model model(model_path);
    auto preds = model(img_path);

    BOOST_REQUIRE(!preds.empty());
}

BOOST_AUTO_TEST_SUITE_END();
