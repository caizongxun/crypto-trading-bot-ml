#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discord Bot - 推理引擎
VM 上推理本地訓練的模簡, 推送 Discord 通知
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

import discord
from discord.ext import commands, tasks
import torch
import numpy as np

from predictor import CryptoPredictor

# ==================== 配置 ====================

load_dotenv()  # 載入 .env 配置

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', '0'))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
predictor = None

# ==================== 箕位函數 ====================


@bot.event
async def on_ready():
    """機器人准備完成"""
    global predictor
    logger.info(f"{bot.user.name} has connected to Discord!")
    
    # 初始化結核推理器
    predictor = CryptoPredictor()
    logger.info("Predictor initialized")
    
    # 啟動定時預測任務
    if not predict_loop.is_running():
        predict_loop.start()
        logger.info("Prediction loop started")


@tasks.loop(minutes=60)  # 每小時推理一次
@tasks.before_loop
async def before_predict_loop():
    await bot.wait_until_ready()


@tasks.loop(minutes=60)
async def predict_loop():
    """定時推理任務"""
    if predictor is None:
        return
    
    try:
        logger.info("Starting prediction cycle...")
        
        # 預測主要幣種
        symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP']
        
        for symbol in symbols:
            try:
                prediction = predictor.predict(symbol)
                
                # 横序 Discord Embed
                embed = discord.Embed(
                    title=f"{symbol} Price Prediction 🔮",
                    description=f"Predicted next hour",
                    color=discord.Color.blue()
                )
                
                embed.add_field(
                    name="Current Price",
                    value=f"${prediction['current_price']:.2f}",
                    inline=True
                )
                
                embed.add_field(
                    name="Predicted Price",
                    value=f"${prediction['predicted_price']:.2f}",
                    inline=True
                )
                
                embed.add_field(
                    name="Change",
                    value=f"{prediction['change_percent']:.2f}%",
                    inline=True
                )
                
                embed.add_field(
                    name="Confidence",
                    value=f"{prediction['confidence']:.1f}%",
                    inline=True
                )
                
                embed.add_field(
                    name="Signal",
                    value=prediction['signal'],
                    inline=True
                )
                
                # 選據批 channel 並讓嬉上 message
                channel = bot.get_channel(DISCORD_CHANNEL_ID)
                if channel:
                    await channel.send(embed=embed)
                    logger.info(f"Sent prediction for {symbol}")
            
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Prediction loop error: {str(e)}")


@bot.command(name='predict')
async def predict_command(ctx, symbol: str):
    """手動預測一個幣種
    用道: !predict SOL
    """
    if predictor is None:
        await ctx.send("Predictor not initialized yet")
        return
    
    try:
        symbol = symbol.upper()
        prediction = predictor.predict(symbol)
        
        # 格式化 response
        message = f"""
        **{symbol} Price Prediction** 🔮
        
Current Price: ${prediction['current_price']:.2f}
Predicted Price: ${prediction['predicted_price']:.2f}
Change: {prediction['change_percent']:.2f}%
Confidence: {prediction['confidence']:.1f}%

Signal: {prediction['signal']}
        """
        
        await ctx.send(message)
    
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
        logger.error(f"Command error: {str(e)}")


@bot.command(name='status')
async def status_command(ctx):
    """雲詳 bot 的狀態
    用道: !status
    """
    if predictor is None:
        await ctx.send("Predictor not ready")
        return
    
    status_msg = f"""
    **Bot Status** 🤖
    
Model Directory: {predictor.model_dir}
Available Models: {len(list(predictor.model_dir.glob('*.pth')))}
Device: {predictor.device}
    """
    
    await ctx.send(status_msg)


@bot.command(name='help')
async def help_command(ctx):
    """
輊侶命令
    用道: !help
    """
    help_msg = """
    **Available Commands** 📜
    
`!predict <symbol>` - 預測一個幣種的下一小時價格
e.g., `!predict SOL`

`!status` - 阻輊 bot 的統計信息

`!help` - 您正在看的
    """
    
    await ctx.send(help_msg)


# ==================== 主程式 ====================


def main():
    """Bot 主程式"""
    logger.info("Starting Discord bot...")
    
    if not DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN not found in .env")
    
    bot.run(DISCORD_TOKEN)


if __name__ == '__main__':
    main()
