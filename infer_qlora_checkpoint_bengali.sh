CUDA_VISIBLE_DEVICES=1 python infer_qlora_checkpoint_bengali.py \
  /home/kawshik/tts-testing/Orpheus-TTS/finetune/checkpoints-bengali-telecom-qlora/checkpoint-1000 \
  "আমাকে বাংলায় কথা বলার জন্য ফাইনটিউন করা হয়েছে। আমাকে কি আপনি ব্যবহার করতে পারবেন?" \
  checkpoint-1000.wav