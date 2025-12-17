#!/usr/bin/env python3
"""
Main entry point for the Local Multimodal AI Agent
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest='command')

    # Add paper command
    add_parser = subparsers.add_parser("add_paper", help="Add and classify a paper")
    add_parser.add_argument("path", type=str, help="Path to the paper")
    add_parser.add_argument("--topics", type=str, required=True,
                            help="Comma separated list of possible topics")

    # Search paper command
    search_paper_parser = subparsers.add_parser("search_paper", help="Search papers by query")
    search_paper_parser.add_argument("query", type=str, help="Query string for semantic search")

    # Search image command
    search_image_parser = subparsers.add_parser("search_image", help="Search images by description")
    search_image_parser.add_argument("query", type=str, help="Description of desired image")

    # Semantic search command
    semantic_search_parser = subparsers.add_parser("semantic_search", help="Fine-grained semantic search with chunks and page numbers")
    semantic_search_parser.add_argument("query", type=str, help="Query string for semantic search")

    # Batch organize command
    batch_organize_parser = subparsers.add_parser("batch_organize", help="Batch organize papers by topics")
    batch_organize_parser.add_argument("source_dir", type=str, help="Directory containing unorganized PDF files")
    batch_organize_parser.add_argument("--topics", type=str, required=True,
                                      help="Comma separated list of topics for classification")

    # Auto organize command (automatic topic discovery)
    auto_organize_parser = subparsers.add_parser("auto_organize", help="Auto-organize papers with automatic topic discovery")
    auto_organize_parser.add_argument("source_dir", type=str, help="Directory containing unorganized PDF files")
    auto_organize_parser.add_argument("--n_clusters", type=int, default=None,
                                     help="Number of topics/clusters (auto-detect if not specified)")

    args = parser.parse_args()

    if args.command == "add_paper":
        from src.document_manager import add_paper
        topics = [topic.strip() for topic in args.topics.split(",")]
        add_paper.process(args.path, topics)
    elif args.command == "search_paper":
        from src.document_manager import search_paper
        search_paper.process(args.query)
    elif args.command == "search_image":
        from src.image_manager import search_image
        search_image.process(args.query)
    elif args.command == "semantic_search":
        from src.document_manager import semantic_search
        semantic_search.process(args.query)
    elif args.command == "batch_organize":
        from src.document_manager import batch_organize
        topics = [topic.strip() for topic in args.topics.split(",")]
        batch_organize.process(args.source_dir, topics)
    elif args.command == "auto_organize":
        from src.document_manager import auto_organize
        auto_organize.process(args.source_dir, args.n_clusters)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()