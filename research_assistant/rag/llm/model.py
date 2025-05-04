"""
Language model initialization and text generation functionality.
"""

import logging
import torch
from typing import Optional, Dict, Any, Union, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)

logger = logging.getLogger(__name__)

class LanguageModel:
    """
    Wrapper for Hugging Face language models for text generation.
    
    This class handles loading, quantization, and generation with 
    state-of-the-art language models.
    """
    
    def __init__(
        self, 
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        quantize: bool = True,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the language model and tokenizer.
        
        Args:
            model_name: Hugging Face model ID or path
            quantize: Whether to apply 4-bit quantization
            device_map: Device mapping strategy ("auto" or specific devices)
            torch_dtype: Data type for model weights
        """
        logger.info(f"Loading language model: {model_name}")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure quantization if enabled
            quantization_config = None
            if quantize:
                logger.info("Using 4-bit quantization for reduced memory usage")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True
                )
            
            # Load model with configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            
            logger.info("Language model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            raise
    
    def generate(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> str:
        """
        Generate text based on the provided prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs. greedy decoding)
            num_return_sequences: Number of generations to return
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text (without the prompt)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Hard limit on input length
        ).to(self.model.device)
        
        # Set up generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the newly generated part (without the prompt)
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()