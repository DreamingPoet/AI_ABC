
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test script.')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, default='output.txt', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='A threshold value')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--names', nargs='+', default=['default'], help='List of names')

    args = parser.parse_args()

    print(f"Executing test script...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Verbose: {args.verbose}")
    print(f"Names: {args.names}")

    # Example of outputting structured data (e.g., JSON)
    result_data = {
        'status': 'success',
        'processed_input': args.input,
        'config': {
            'output_file': args.output,
            'threshold': args.threshold,
            'verbose_mode': args.verbose,
            'names_list': args.names
        }
    }
    print("---JSON START---")
    print(json.dumps(result_data, indent=2))
    print("---JSON END---")

    # You could also write to stderr
    # import sys
    # print("This is a test error message.", file=sys.stderr)
