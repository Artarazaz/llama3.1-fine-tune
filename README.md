# Fine-Tuning Llama 3.1-8B for Simulating Narcissistic Speech Patterns

## Project Overview

This project fine-tunes the large language model **Meta-Llama-3.1-8B-Instruct** using the **LoRA** (Low-Rank Adaptation) method. The primary goal is to simulate the characteristic speech and behavioral patterns associated with individuals exhibiting features of Narcissistic Personality Disorder (NPD).

A custom dataset consisting of approximately 5,800 Persian conversation samples in the system–user–assistant format was specifically designed and created. The assistant responses were crafted to reflect core narcissistic traits: grandiosity, justification of inappropriate behaviors, lack of empathy, and excessive admiration/praise.

Training was performed over only two epochs, with results demonstrating successful learning of the target behavioral and linguistic patterns.

## Project Objectives

- Simulate narcissistic speech patterns using large language models  
- Evaluate the effectiveness of the LoRA method for domain-specific fine-tuning of large models  
- Create a structured Persian dataset suitable for this task  
- Perform quantitative and qualitative evaluation of the fine-tuned model

## Main Technical Specifications

- Base model: Meta-Llama-3.1-8B-Instruct  
- Fine-tuning method: LoRA (rank=8, alpha=16, target_modules: q_proj, v_proj)  
- Training samples: ~4,650  
- Epochs: 2  
- Maximum sequence length: 512 tokens  
- Optimizer: AdamW  
- Mixed precision training (FP16)  
- Gradient accumulation and gradient checkpointing for memory efficiency

## Intended Applications

- Research in computational psychology  
- Educational tool for students and mental health professionals  
- Study of linguistic patterns associated with personality disorders  
- Foundation for developing automated text-based personality pattern analysis tools

## Important Notes

This project has been developed strictly for **research and educational purposes** in the field of computational psychology.  

It must **never** be used as a clinical diagnostic tool, therapeutic advice system, or substitute for qualified psychologists or psychiatrists.

## License

Apache 2.0  
(Base model remains subject to the Meta Llama 3.1 Community License)

---

University of Bojnord – Faculty of Engineering – Department of Computer Engineering  
Academic Year 2024–2025

# فاین‌تیونینگ Llama 3.1-8B برای شبیه‌سازی الگوهای گفتاری نارسیستی

## چکیده پروژه

این پروژه به فاین‌تیونینگ مدل زبانی بزرگ **Meta-Llama-3.1-8B-Instruct** با استفاده از روش **LoRA** (Low-Rank Adaptation) می‌پردازد. هدف اصلی، شبیه‌سازی الگوهای گفتاری و رفتاری مشخصه افراد دارای ویژگی‌های اختلال شخصیت نارسیستی (Narcissistic Personality Disorder - NPD) است.

مجموعه داده اختصاصی شامل حدود ۵۸۰۰ نمونه مکالمه فارسی در قالب system-user-assistant طراحی و تولید شده است. این داده‌ها به‌گونه‌ای ساخته شده‌اند که پاسخ‌های assistant ویژگی‌های کلیدی نارسیستی شامل خودبزرگ‌بینی، توجیه رفتارهای نادرست، فقدان همدلی و تمجید مفرط را نشان دهند.

آموزش با تنها دو دور (epoch) انجام شده و نتایج نشان‌دهنده یادگیری موفق الگوهای مورد نظر است.

## اهداف پروژه

- شبیه‌سازی الگوهای گفتاری نارسیستی با استفاده از مدل‌های زبانی بزرگ
- بررسی کارایی روش LoRA در فاین‌تیونینگ تخصصی مدل‌های بزرگ
- تولید مجموعه داده ساختاریافته فارسی برای این کاربرد
- ارزیابی کمی و کیفی عملکرد مدل فاین‌تیون‌شده

## مشخصات فنی اصلی

- مدل پایه: Meta-Llama-3.1-8B-Instruct  
- روش: LoRA (rank=8, alpha=16, target_modules: q_proj, v_proj)  
- تعداد نمونه‌های آموزشی: ~۴۶۵۰  
- تعداد epoch: ۲  
- حداکثر طول دنباله: ۵۱۲ توکن  
- بهینه‌ساز: AdamW  
- یادگیری ترکیبی (Mixed Precision FP16)  
- Gradient Accumulation و Gradient Checkpointing برای مدیریت حافظه

## کاربردهای مورد نظر

- تحقیقات در روان‌شناسی محاسباتی  
- ابزار آموزشی برای دانشجویان و درمانگران حوزه سلامت روان  
- مطالعه الگوهای زبانی مرتبط با اختلالات شخصیتی  
- پایه‌ای برای توسعه ابزارهای تحلیل و تشخیص خودکار الگوهای شخصیتی از متن

## ملاحظات مهم

این پروژه صرفاً با اهداف تحقیقاتی و آموزشی توسعه یافته است و **نباید** به هیچ عنوان به عنوان ابزار تشخیص بالینی، مشاوره درمانی یا جایگزین متخصص روان‌شناسی/روان‌پزشکی استفاده شود.

## لایسنس

Apache 2.0  
(مدل پایه مشمول Llama 3.1 Community License است)

---

دانشگاه بجنورد – دانشکده فنی و مهندسی – گروه مهندسی کامپیوتر  
سال تحصیلی ۱۴۰۳–۱۴۰۴
