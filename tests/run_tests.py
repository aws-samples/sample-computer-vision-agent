#!/usr/bin/env python3
"""
Test runner for the CV MCP Server tests.

This script runs the CV tools test suite and provides a report
of the test results for the computer vision functionality.
"""

import logging
import os
import sys
import unittest

# Configure logging to reduce noise during testing
logging.basicConfig(level=logging.ERROR)

# Add application directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))


def run_all_tests():
    """Run CV tools test suite and return results"""

    print("Running CV MCP Server Tests...")
    print("This will test:")
    print("1. Computer vision tools functionality")
    print("2. S3 integration")
    print("3. Image processing operations")
    print("4. Error handling scenarios")
    print()
    print("-" * 60)

    # Discover and run only the CV tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_cv_tools.py")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result


def print_test_summary(result):
    """Print a summary of test results"""
    print("\n" + "=" * 60)
    print("CV MCP SERVER TEST SUMMARY")
    print("=" * 60)

    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            newline = "\n"
            print(
                f"- {test}: {traceback.split('AssertionError: ')[-1].split(newline)[0]}"
            )

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            newline = "\n"
            print(f"- {test}: {traceback.split(newline)[-2]}")

    success_rate = (
        (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            * 100
        )
        if result.testsRun > 0
        else 0
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        print("The CV MCP Server is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the failures and errors above.")

    print("=" * 60)


def main():
    """Main test runner function"""
    try:
        result = run_all_tests()
        print_test_summary(result)

        # Return appropriate exit code
        return 0 if result.wasSuccessful() else 1

    except Exception as e:
        print(f"\nFATAL ERROR: Failed to run tests: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
