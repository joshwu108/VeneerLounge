"""
Flask API server for veneer preview generation.
This server provides REST endpoints for:
- Generating veneer previews
- Health checks
- Model information
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import base64
import io
from pathlib import Path
from PIL import Image
import traceback
import logging

sys.path.insert(0, str(Path(__file__).parent))
from veneer_service import get_veneer_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

_service = None


def get_service():
    """Get or initialize the veneer service."""
    global _service
    if _service is None:
        import os
        model_type = os.environ.get('VENEER_MODEL_TYPE', 'sdxl')

        if model_type == 'sdxl':
            # SDXL needs no special config
            config = {}
        elif model_type == 'controlnet':
            controlnet_path = os.environ.get(
                'CONTROLNET_PATH',
                'lllyasviel/control_v11p_sd15_seg'
            )
            seg_checkpoint_path = Path(__file__).parent.parent.parent / \
                                'ext/individual_tooth_segmentation/checkpoints/CP_teeth_seg.pth'

            config = {
                'controlnet_path': controlnet_path,
            }
            if seg_checkpoint_path.exists():
                config['segmentation_checkpoint'] = str(seg_checkpoint_path)
                print(f"Using tooth segmentation checkpoint: {seg_checkpoint_path}")
            else:
                print("Warning: Tooth segmentation checkpoint not found. Using fallback method.")
                print(f"Expected at: {seg_checkpoint_path}")
                # ControlNet will use fallback color-based segmentation
                config['segmentation_checkpoint'] = None
        else:
            config = {
                'checkpoint_path': str(Path(__file__).parent.parent.parent /
                                      'ext/veneer_generation/checkpoints/pix2pix_veneer/best.pth')
            }

        _service = get_veneer_service(model_type=model_type, **config)

    return _service


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON with status and model information
    """
    try:
        service = get_service()
        return jsonify({
            'status': 'healthy',
            'model_type': service.model_type,
            'device': str(service.device)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/veneer-preview', methods=['POST'])
def generate_preview():
    """
    Generate veneer preview from base64 encoded image.
    """
    try:
        logger.info("=== Starting veneer preview generation ===")
        data = request.json

        if not data or 'image' not in data:
            logger.error("Missing 'image' field in request")
            return jsonify({
                'error': 'Missing required field: image',
                'success': False
            }), 400

        logger.info("Step 1: Request received, parsing parameters")

        image_base64 = data['image']
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        intensity = data.get('intensity', 0.75)
        preserve_geometry = data.get('preserve_geometry', False)
        custom_prompt = data.get('custom_prompt', None)
        bounding_box = data.get('bounding_box', None)

        if not 0 <= intensity <= 1:
            logger.error(f"Invalid intensity value: {intensity}")
            return jsonify({
                'error': 'Intensity must be between 0 and 1',
                'success': False
            }), 400

        logger.info("Step 2: Getting service instance")
        service = get_service()

        logger.info("Step 3: Calling generate_from_base64")
        # Generate preview
        preview_base64 = service.generate_from_base64(
            base64_image=image_base64,
            intensity=intensity,
            preserve_geometry=preserve_geometry,
            custom_prompt=custom_prompt,
            bounding_box=bounding_box
        )
        logger.info("Step 4: Generation complete, returning result")
        return jsonify({
            'output': [preview_base64],
            'success': True
        })

    except Exception as e:
        logger.error("="*60)
        logger.error(f"ERROR generating preview: {e}")
        logger.error("="*60)
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())  # This logs to logger instead of print
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None,  # Only in debug mode
            'success': False
        }), 500



@app.route('/api/veneer-preview/file', methods=['POST'])
def generate_preview_file():
    """
    Generate veneer preview from uploaded file.

    Request:
        Multipart form data with 'image' file field

    Returns:
        {
            "output": ["data:image/jpeg;base64,..."],
            "success": true
        }
    """
    try:
        # Check for file
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400

        file = request.files['image']

        # Read image
        image = Image.open(file.stream).convert('RGB')

        # Get parameters
        intensity = float(request.form.get('intensity', 0.75))
        preserve_geometry = request.form.get('preserve_geometry', 'false').lower() == 'true'
        custom_prompt = request.form.get('custom_prompt', None)

        # Get service
        service = get_service()

        # Generate preview
        preview = service.generate_from_pil(
            image=image,
            intensity=intensity,
            preserve_geometry=preserve_geometry,
            custom_prompt=custom_prompt
        )

        # Convert to base64
        buffer = io.BytesIO()
        preview.save(buffer, format='JPEG', quality=95)
        preview_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        preview_base64 = f"data:image/jpeg;base64,{preview_base64}"

        return jsonify({
            'output': [preview_base64],
            'success': True
        })

    except Exception as e:
        print(f"Error generating preview from file: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.

    Returns:
        JSON with model information
    """
    try:
        service = get_service()
        return jsonify({
            'model_type': service.model_type,
            'device': str(service.device),
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """
    Reload the model (useful for switching models).

    Request body:
        {
            "model_type": "controlnet" or "pix2pix"
        }

    Returns:
        {
            "success": true,
            "model_type": "..."
        }
    """
    global _service
    try:
        data = request.json
        model_type = data.get('model_type', 'controlnet')

        if model_type not in ['sdxl', 'controlnet', 'pix2pix']:
            return jsonify({
                'error': 'Invalid model_type. Must be "sdxl", "controlnet" or "pix2pix"',
                'success': False
            }), 400

        # Clear current service
        _service = None

        # Set environment variable
        import os
        os.environ['VENEER_MODEL_TYPE'] = model_type

        # Reload
        service = get_service()

        return jsonify({
            'success': True,
            'model_type': service.model_type,
            'device': str(service.device)
        })

    except Exception as e:
        print(f"Error reloading model: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Veneer Preview API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to bind to')
    parser.add_argument('--model', type=str, default='sdxl',
                       choices=['sdxl', 'controlnet', 'pix2pix'],
                       help='Model type to use')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')

    args = parser.parse_args()

    # Set model type
    import os
    os.environ['VENEER_MODEL_TYPE'] = args.model

    # Configure Flask logging
    if args.debug:
        logging.getLogger('werkzeug').setLevel(logging.DEBUG)
        app.logger.setLevel(logging.DEBUG)
    else:
        logging.getLogger('werkzeug').setLevel(logging.INFO)
        app.logger.setLevel(logging.INFO)

    print("=" * 60)
    print("Veneer Preview API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")
    print(f"Debug: {args.debug}")
    print()
    print("Endpoints:")
    print(f"  POST   /api/veneer-preview        - Generate preview from base64")
    print(f"  POST   /api/veneer-preview/file   - Generate preview from file")
    print(f"  GET    /api/model/info            - Get model information")
    print(f"  POST   /api/model/reload          - Reload model")
    print(f"  GET    /health                    - Health check")
    print()
    print(f"Starting server...")
    print("=" * 60)
    print()

    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )