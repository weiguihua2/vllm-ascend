curl http://0.0.0.0:10086/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "prompt": ["The president of United States is","The president of United States is","The president of United States is","The president of United States is"],
    "max_tokens": 100,
    "temperature": 0
  }'
