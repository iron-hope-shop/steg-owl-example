import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import struct
import zlib
import base64

def generate_gradient_image(width, height):
    # Create a horizontal gradient image (RGB)
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    # Stack for R, G, B channels (making a grayscale gradient image in color)
    img_array = np.stack((gradient, gradient, gradient), axis=-1)
    return Image.fromarray(img_array, 'RGB')

def text_to_bits(text):
    # Convert text to a binary string (each char -> 8 bits)
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    # Convert binary string back to text (each 8 bits -> 1 char)
    # Break into 8-bit chunks
    text = ""
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i:i+8]
            try:
                text += chr(int(byte, 2))
            except ValueError:
                # Skip invalid bytes
                pass
    return text

def embed_payload(image, payload, use_all_channels=False):
    """
    Embed payload (binary string) into the 2 least significant bits of the channels.
    If use_all_channels is True, use all RGB channels, otherwise just blue.
    
    Process:
    1. Compress the payload using zlib
    2. Add a CRC32 checksum
    3. Embed with length header
    """
    # Convert image to numpy array
    img_data = np.array(image)
    h, w, c = img_data.shape
    
    # Calculate capacity
    if use_all_channels:
        # We can use 2 bits per channel (6 bits per pixel)
        capacity = h * w * c * 2
    else:
        # Just using blue channel (2 bits per pixel)
        capacity = h * w * 2
    
    # Prepare payload with compression and error detection
    try:
        # Convert to bytes if it's a string
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload_bytes = payload
            
        # Compress the payload
        compressed_payload = zlib.compress(payload_bytes, level=9)
        
        # Calculate CRC32 checksum (4 bytes)
        crc = zlib.crc32(compressed_payload) & 0xffffffff
        crc_bytes = crc.to_bytes(4, byteorder='big')
        
        # Create header with compressed payload length (4 bytes)
        length = len(compressed_payload)
        length_bytes = length.to_bytes(4, byteorder='big')
        
        # Combine everything
        full_payload_bytes = length_bytes + crc_bytes + compressed_payload
        
        # Convert to binary string
        binary_payload = ''.join(format(b, '08b') for b in full_payload_bytes)
        
        print(f"Original size: {len(payload_bytes)} bytes")
        print(f"Compressed size: {len(compressed_payload)} bytes")
        print(f"Compression ratio: {len(compressed_payload)/len(payload_bytes):.2%}")
        print(f"Total data size with headers: {len(full_payload_bytes)} bytes")
    
    except Exception as e:
        print(f"Error preparing payload: {e}")
        raise
    
    # Check if the payload fits in the image
    if len(binary_payload) > capacity:
        raise ValueError(f"Payload is too large for this image. Capacity: {capacity} bits, Payload: {len(binary_payload)} bits")
    
    # Flatten the image data if using all channels
    if use_all_channels:
        flattened_data = img_data.reshape(-1)
        new_data = np.copy(flattened_data)
        
        # For each color value in the flattened image
        bit_index = 0
        for i in range(len(flattened_data)):
            if bit_index >= len(binary_payload):
                break
                
            # Get next 2 bits from payload (or just 1 if at the end)
            if bit_index + 1 < len(binary_payload):
                bits = binary_payload[bit_index:bit_index+2]
                bit_index += 2
            else:
                bits = binary_payload[bit_index] + '0'
                bit_index += 1
                
            # Clear the 2 LSBs and then set them to the payload bits
            new_data[i] = (flattened_data[i] & 0xFC) | int(bits, 2)
                
        # Reshape back to the original image shape
        img_data = new_data.reshape((h, w, c))
        
    else:
        # Original method: Embed only in blue channel
        blue_channel = img_data[:, :, 2].flatten()
        new_blue = np.copy(blue_channel)
        
        bit_index = 0
        for i in range(len(blue_channel)):
            if bit_index >= len(binary_payload):
                break
                
            # Get next 2 bits from payload (or just 1 if at the end)
            if bit_index + 1 < len(binary_payload):
                bits = binary_payload[bit_index:bit_index+2]
                bit_index += 2
            else:
                bits = binary_payload[bit_index] + '0'
                bit_index += 1
                
            # Clear the 2 LSBs and then set them to the payload bits
            new_blue[i] = (blue_channel[i] & 0xFC) | int(bits, 2)
        
        # Replace blue channel in the image data
        img_data[:, :, 2] = new_blue.reshape((h, w))
    
    return Image.fromarray(img_data, 'RGB')

def extract_payload(stego_image, use_all_channels=False):
    """
    Extract message from the 2 LSBs of the color channels.
    If use_all_channels is True, extract from all RGB channels, otherwise just blue.
    
    Process:
    1. Extract bits from image
    2. Read length header (4 bytes)
    3. Read CRC32 checksum (4 bytes)
    4. Extract and verify compressed payload
    5. Decompress and return
    """
    # Convert image to numpy array
    img_data = np.array(stego_image)
    h, w, c = img_data.shape
    
    # Extract bits
    if use_all_channels:
        # Extract from all channels
        flattened_data = img_data.reshape(-1)
        extracted_bits = ''
        
        # First extract enough bits for headers (8 bytes = 64 bits)
        min_bits_needed = 64
        bits_collected = 0
        
        for pixel_val in flattened_data:
            extracted_bits += format(pixel_val & 0b11, '02b')
            bits_collected += 2
            
            if bits_collected >= min_bits_needed:
                break
                
        # Parse the length from the first 4 bytes (32 bits)
        length_bits = extracted_bits[:32]
        length_bytes = bytes([int(length_bits[i:i+8], 2) for i in range(0, 32, 8)])
        payload_length = int.from_bytes(length_bytes, byteorder='big')
        
        # Parse the CRC32 from the next 4 bytes
        crc_bits = extracted_bits[32:64]
        crc_bytes = bytes([int(crc_bits[i:i+8], 2) for i in range(0, 32, 8)])
        expected_crc = int.from_bytes(crc_bytes, byteorder='big')
        
        # Calculate how many more bits we need to extract
        payload_bits_needed = payload_length * 8
        total_bits_needed = 64 + payload_bits_needed
        
        # Continue extracting remaining bits
        for i in range(bits_collected // 2, min(len(flattened_data), (total_bits_needed + 1) // 2)):
            pixel_val = flattened_data[i]
            extracted_bits += format(pixel_val & 0b11, '02b')
            
            if len(extracted_bits) >= total_bits_needed:
                break
            
    else:
        # Extract only from blue channel
        blue_channel = img_data[:, :, 2].flatten()
        extracted_bits = ''
        
        # First extract enough bits for headers (8 bytes = 64 bits)
        min_bits_needed = 64
        bits_collected = 0
        
        for pixel in blue_channel:
            extracted_bits += format(pixel & 0b11, '02b')
            bits_collected += 2
            
            if bits_collected >= min_bits_needed:
                break
        
        # Parse the length from the first 4 bytes (32 bits)
        length_bits = extracted_bits[:32]
        length_bytes = bytes([int(length_bits[i:i+8], 2) for i in range(0, 32, 8)])
        payload_length = int.from_bytes(length_bytes, byteorder='big')
        
        # Parse the CRC32 from the next 4 bytes
        crc_bits = extracted_bits[32:64]
        crc_bytes = bytes([int(crc_bits[i:i+8], 2) for i in range(0, 32, 8)])
        expected_crc = int.from_bytes(crc_bytes, byteorder='big')
        
        # Calculate how many more bits we need to extract
        payload_bits_needed = payload_length * 8
        total_bits_needed = 64 + payload_bits_needed
        
        # Continue extracting remaining bits
        for i in range(bits_collected // 2, min(len(blue_channel), (total_bits_needed + 1) // 2)):
            pixel = blue_channel[i]
            extracted_bits += format(pixel & 0b11, '02b')
            
            if len(extracted_bits) >= total_bits_needed:
                break
    
    # Ensure we extracted enough bits
    if len(extracted_bits) < 64:
        return "Error: Not enough bits extracted for headers", False, 0
    
    # Extract the compressed payload
    payload_bits = extracted_bits[64:64 + payload_bits_needed]
    
    # Check if we got enough bits
    if len(payload_bits) < payload_bits_needed:
        print(f"Warning: Could only extract {len(payload_bits)} of {payload_bits_needed} required bits")
        
    # Convert bits to bytes
    payload_bytes = bytearray()
    for i in range(0, len(payload_bits), 8):
        if i + 8 <= len(payload_bits):
            byte = payload_bits[i:i+8]
            payload_bytes.append(int(byte, 2))
    
    # Verify CRC32
    actual_crc = zlib.crc32(payload_bytes) & 0xffffffff
    crc_match = (actual_crc == expected_crc)
    
    try:
        # Decompress the payload
        decompressed_data = zlib.decompress(payload_bytes)
        
        # Convert bytes to text
        extracted_text = decompressed_data.decode('utf-8')
        
        return extracted_text, crc_match, len(decompressed_data)
    except Exception as e:
        print(f"Error decompressing data: {e}")
        return f"Error decompressing data: {e}", False, len(payload_bytes)

def create_diff_image(original, stego):
    """Create a visualization of the differences between original and stego images"""
    # Convert to numpy arrays
    orig_array = np.array(original)
    stego_array = np.array(stego)
    
    # Calculate absolute difference
    diff = np.abs(orig_array - stego_array)
    
    # Create a heatmap for visualization (scaling up differences)
    diff_scaled = diff * 20  # Scale for better visibility
    diff_scaled = np.clip(diff_scaled, 0, 255).astype(np.uint8)
    
    # Create a difference heatmap using all channels
    plt.figure(figsize=(10, 8))
    
    # Calculate the average of the differences across all channels
    avg_diff = np.mean(diff, axis=2)
    plt.imshow(avg_diff, cmap='hot')
    plt.colorbar(label='Difference Value')
    plt.title('Color Channel Differences (Steganography Location)')
    
    return Image.fromarray(diff_scaled), plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Image Steganography Tool')
    parser.add_argument('-i', '--image', help='Path to input image (optional)')
    parser.add_argument('-m', '--message', help='Message to embed in the image')
    parser.add_argument('-f', '--file', help='Path to text file containing message to embed')
    parser.add_argument('-e', '--extract', action='store_true', 
                       help='Extract message from the stego image instead of embedding')
    parser.add_argument('-a', '--all-channels', action='store_true',
                       help='Use all color channels (RGB) for steganography (increases capacity)')
    parser.add_argument('-o', '--output', help='Output directory name (defaults to timestamped directory)')
    args = parser.parse_args()
    
    # Create a unique output directory based on timestamp or user-specified name
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"stego_output_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.extract:
        # Extract mode
        if not args.image:
            print("Error: Please provide a stego image to extract from")
            return
        
        # Open the image
        try:
            stego_img = Image.open(args.image)
            print(f"Reading stego image: {args.image}")
        except Exception as e:
            print(f"Error opening image: {e}")
            return
        
        # Extract the payload
        try:
            print("Extracting hidden message...")
            extracted_text, checksum_verified, orig_length = extract_payload(stego_img, use_all_channels=args.all_channels)
            
            # Show status
            verification_status = "VERIFIED ✓" if checksum_verified else "FAILED ✗"
            print(f"CRC32 checksum: {verification_status}")
            print(f"Original message length: {orig_length} bytes")
            print(f"Extracted message length: {len(extracted_text)} characters")
            
            # Save the extracted message
            extract_file = os.path.join(output_dir, "extracted_message.txt")
            with open(extract_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                
            # Print a preview
            preview_length = min(200, len(extracted_text))
            print(f"\nMessage preview: '{extracted_text[:preview_length]}{'...' if len(extracted_text) > preview_length else ''}'")
            print(f"\nFull message saved to: {extract_file}")
            
            # Save metadata
            meta_file = os.path.join(output_dir, "extraction_info.txt")
            with open(meta_file, 'w') as f:
                f.write(f"Steganography Extraction - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Used all channels: {'Yes' if args.all_channels else 'No'}\n")
                f.write(f"Image source: {args.image}\n")
                f.write(f"CRC32 verification: {verification_status}\n")
                f.write(f"Original message length: {orig_length} bytes\n")
                f.write(f"Extracted message length: {len(extracted_text)} characters\n")
            
            print(f"Extraction info saved to: {meta_file}")
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            
        return
    
    # Embedding mode
    # Either load the provided image or generate a gradient
    if args.image:
        try:
            original_img = Image.open(args.image)
            print(f"Using provided image: {args.image}")
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Falling back to generated gradient image")
            original_img = generate_gradient_image(800, 600)
    else:
        print("No image provided, generating gradient image")
        original_img = generate_gradient_image(800, 600)
    
    # Get message either from command line or file
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                payload = f.read()
            print(f"Using message from file: {args.file}")
        except Exception as e:
            print(f"Error reading message file: {e}")
            return
    elif args.message:
        payload = args.message
    else:
        payload = "The Universe in 2 Bits: 42 & Beyond"
        print(f"No message provided, using default message")
    
    # Ensure the image is in RGB mode
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
    
    # Calculate capacity
    h, w, c = np.array(original_img).shape
    if args.all_channels:
        capacity = h * w * c * 2
    else:
        capacity = h * w * 2
        
    payload_bytes = len(payload.encode('utf-8'))
    print(f"Message length: {len(payload)} characters ({payload_bytes} bytes)")
    
    # Embed payload into the image
    try:
        print("Embedding message...")
        stego_img = embed_payload(original_img, payload, use_all_channels=args.all_channels)
        print("Message embedded successfully!")
        
        # Calculate difference visualization
        print("Generating visualizations...")
        diff_img, diff_heatmap = create_diff_image(original_img, stego_img)
        
        # Save images
        original_path = os.path.join(output_dir, "original.png")
        stego_path = os.path.join(output_dir, "stego_message.png")
        diff_path = os.path.join(output_dir, "difference.png")
        heatmap_path = os.path.join(output_dir, "difference_heatmap.png")
        info_path = os.path.join(output_dir, "info.txt")
        
        original_img.save(original_path)
        stego_img.save(stego_path, format='PNG', compress_level=0)  # Use no compression for PNG
        diff_img.save(diff_path)
        diff_heatmap.savefig(heatmap_path)
        plt.close()
        
        # Create magnified view of original vs. stego
        magnify_area = (100, 100, 200, 200)  # Example area to magnify
        original_crop = original_img.crop(magnify_area)
        stego_crop = stego_img.crop(magnify_area)
        
        # Create side-by-side comparison
        comparison = Image.new('RGB', (original_crop.width * 2, original_crop.height))
        comparison.paste(original_crop, (0, 0))
        comparison.paste(stego_crop, (original_crop.width, 0))
        comparison_path = os.path.join(output_dir, "magnified_comparison.png")
        comparison.save(comparison_path)
        
        # Save message as a separate file for reference
        message_path = os.path.join(output_dir, "embedded_message.txt")
        with open(message_path, 'w', encoding='utf-8') as f:
            f.write(payload)
        
        # Verify the embedded message by extracting it
        print("Verifying embedded message...")
        verification_text, checksum_verified, orig_length = extract_payload(stego_img, use_all_channels=args.all_channels)
        verification_status = "PASSED ✓" if checksum_verified else "FAILED ✗"
        match_percentage = sum(1 for a, b in zip(verification_text, payload) if a == b) / len(payload) * 100 if len(payload) > 0 else 0
        
        # Save information text file
        with open(info_path, 'w') as f:
            f.write(f"Steganography Output - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Used all channels: {'Yes' if args.all_channels else 'No'}\n")
            if len(payload) > 100:
                f.write(f"Embedded message: '{payload[:100]}...' (truncated, {len(payload)} chars total)\n")
            else:
                f.write(f"Embedded message: '{payload}'\n")
            f.write(f"Message length: {payload_bytes} bytes\n")
            f.write(f"Image dimensions: {w}x{h}\n")
            f.write(f"Steganography capacity: {capacity} bits ({capacity // 8} bytes)\n")
            f.write(f"\nVerification status: {verification_status}\n")
            f.write(f"Verified length: {len(verification_text)} characters\n")
            f.write(f"Content match: {match_percentage:.2f}%\n")
            f.write(f"CRC32 verified: {'Yes' if checksum_verified else 'No'}\n")
            f.write(f"\nTo extract the message, run: python3 main.py -e -i {stego_path} -a -o extract_output")
        
        print(f"\nAll output saved to directory: {output_dir}")
        print(f"Original image: {original_path}")
        print(f"Stego image with embedded message: {stego_path}")
        print(f"Difference visualization: {diff_path}")
        print(f"Difference heatmap: {heatmap_path}")
        print(f"Magnified comparison: {comparison_path}")
        print(f"Information saved to: {info_path}")
        print(f"Embedded message saved to: {message_path}")
        print(f"Verification: {verification_status}")
        print(f"Content match: {match_percentage:.2f}%")
        print(f"\nTo extract the message: python3 main.py -e -i {stego_path} -a -o extract_output")
        
    except ValueError as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()
