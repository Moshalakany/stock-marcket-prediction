import uvicorn
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Start Stock News Sentiment Analysis API server')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server to')
    
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind the server to')
    
    parser.add_argument('--model_dir', type=str, 
                        default=r'E:\Stock Market GP\Reddit\news\sentiment_model_output\enhanced_hybrid_model',
                        help='Directory containing the sentiment model and tokenizer')
    
    parser.add_argument('--data_dir', type=str, 
                        default=r'E:\Stock Market GP\Reddit\news\sentiment_data',
                        help='Directory containing sentiment data')
    
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload on code changes')
    
    parser.add_argument('--preload_model', action='store_true', default=True,
                        help='Preload the model at server startup (default: True)')
    
    args = parser.parse_args()
    
    # Set environment variables for the API
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["DATA_DIR"] = args.data_dir
    os.environ["PRELOAD_MODEL"] = "1" if args.preload_model else "0"
    
    print(f"Starting Sentiment API server on {args.host}:{args.port}")
    print(f"Model directory: {args.model_dir}")
    print(f"Preloading model: {'Yes' if args.preload_model else 'No'}")
    
    uvicorn.run(
        "sentiment_api:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )

if __name__ == "__main__":
    main()
