#!/usr/bin/env python3

import click
from vision import VisionAnalyzer

@click.group()
def cli():
    """AI-powered screenshot analysis tool."""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def analyze(image_path):
    """Analyze the contents of a screenshot or image file.
    
    IMAGE_PATH: Path to the image file to analyze
    """
    try:
        # Initialize analyzer
        analyzer = VisionAnalyzer()

        # Analyze the image
        click.echo("Analyzing image...")
        description = analyzer.analyze_image(image_path)

        # Display results
        click.echo("\nAnalysis Results:")
        click.echo("----------------")
        click.echo(description)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 