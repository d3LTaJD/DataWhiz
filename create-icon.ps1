# Script to convert PNG to ICO for Windows icon
# This uses PowerShell and .NET to create an icon file

param(
    [string]$InputFile = "mind-map.png",
    [string]$OutputFile = "assets/icon.ico"
)

Write-Host "Converting $InputFile to $OutputFile..." -ForegroundColor Yellow

# Load required assemblies
Add-Type -AssemblyName System.Drawing

try {
    # Load the image
    $bitmap = [System.Drawing.Bitmap]::FromFile((Resolve-Path $InputFile))
    
    # Create icon sizes (Windows needs multiple sizes in ICO)
    $sizes = @(256, 128, 64, 48, 32, 16)
    
    # Create a list of icon images
    $iconImages = New-Object System.Collections.ArrayList
    
    foreach ($size in $sizes) {
        # Resize image
        $resized = New-Object System.Drawing.Bitmap($size, $size)
        $graphics = [System.Drawing.Graphics]::FromImage($resized)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.DrawImage($bitmap, 0, 0, $size, $size)
        
        $iconImages.Add($resized) | Out-Null
        $graphics.Dispose()
    }
    
    # Save as ICO using a workaround (PowerShell can't directly save ICO)
    # We'll save as PNG first and use a converter, or just save the largest size
    
    # For now, save the 256x256 version as a temporary solution
    # Note: This creates a basic ICO structure
    Write-Host "Creating icon file..." -ForegroundColor Yellow
    
    # Simple approach: Use the 256x256 image for now
    # You might want to use an online converter or ImageMagick for proper multi-size ICO
    $bitmap256 = New-Object System.Drawing.Bitmap(256, 256)
    $graphics256 = [System.Drawing.Graphics]::FromImage($bitmap256)
    $graphics256.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics256.DrawImage($bitmap, 0, 0, 256, 256)
    
    # Save as PNG first (we'll need to convert manually or use a tool)
    $pngPath = "assets/icon_temp.png"
    $bitmap256.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    
    Write-Host "" -ForegroundColor Green
    Write-Host "Temporary PNG created: $pngPath" -ForegroundColor Green
    Write-Host "" -ForegroundColor Yellow
    Write-Host "IMPORTANT: To create a proper .ico file:" -ForegroundColor Yellow
    Write-Host "1. Use an online converter: https://convertio.co/png-ico/" -ForegroundColor White
    Write-Host "2. Or use ImageMagick: magick convert $pngPath $OutputFile" -ForegroundColor White
    Write-Host "3. Upload $pngPath and download as .ico" -ForegroundColor White
    Write-Host ""
    
    $bitmap.Dispose()
    $bitmap256.Dispose()
    $graphics256.Dispose()
    
    foreach ($img in $iconImages) {
        $img.Dispose()
    }
    
    Write-Host "PNG file ready for conversion at: $pngPath" -ForegroundColor Green
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Use an online PNG to ICO converter:" -ForegroundColor Yellow
    Write-Host "1. Go to https://convertio.co/png-ico/ or https://www.icoconverter.com/" -ForegroundColor White
    Write-Host "2. Upload $InputFile" -ForegroundColor White
    Write-Host "3. Download as icon.ico" -ForegroundColor White
    Write-Host "4. Save it to assets/icon.ico" -ForegroundColor White
}

