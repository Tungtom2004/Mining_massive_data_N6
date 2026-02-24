Mining_massive_data_N6
Project Overview

This project analyzes behavioral patterns of political Telegram channels during the 2024 U.S. election using large-scale message data.

Instead of focusing on individual users, we treat each Telegram channel as a communication entity. Since Telegram channels are primarily admin-driven (only admins can post), we analyze how each channel behaves in terms of:

Activity level

Content strategy

Influence and engagement

Toxicity of content

The goal is to understand how different channels operate within the political information ecosystem on Telegram.

Dataset

We use the following dataset:

US Election 2024 Telegram Distilled Dataset
🔗 https://huggingface.co/datasets/leonardoblas/us_election_2024_telegram_distilled

This dataset contains large-scale Telegram messages related to the 2024 U.S. election, including:

Message content

Timestamps

Views, forwards, replies

Toxicity scores (for English messages)

Hashtags and shared domains

Metadata about channels

The dataset allows us to analyze communication dynamics at scale.

Project Objectives

The main objectives of this project are:

Characterize channel behavior using aggregated behavioral features.

Identify distinct behavioral archetypes through clustering.

Analyze how activity, influence, and toxicity are distributed across channels.

Explore the relationship between toxicity and engagement.

Rather than analyzing political ideology, this project focuses on structural and behavioral dynamics of large-scale communication.

Behavioral Features

Each Telegram channel is represented as a behavioral profile constructed from aggregated statistics such as:

Posting frequency and activity duration

Ratio of original posts vs. forwarded posts

Average views, forwards, and replies

Toxicity scores (English subset)

Content diversity (domains and hashtags)

These features are used to identify patterns and differences between channels.

Expected Insights

We aim to uncover:

Whether channel activity and influence follow heavy-tailed distributions

Whether a small fraction of channels dominates information production

Whether distinct behavioral archetypes emerge (e.g., broadcasters, amplifiers, high-toxicity channels)

Whether toxicity is associated with higher engagement

This analysis helps reveal how information is produced, amplified, and interacted with at scale.
