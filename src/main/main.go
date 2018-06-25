// go run main.go 0 [modelfile] [descriptionsfile]

package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"os"
	"strconv"
	"gocv.io/x/gocv"
)

func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return lines, scanner.Err()
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: main [camera ID] [modelfile] [descriptionsfile]")
		return
	}

	cameraID, _ := strconv.Atoi(os.Args[1])
	model := os.Args[2]
	descr := os.Args[3]
	descriptions, err := readDescriptions(descr)
	if err != nil {
		fmt.Printf("Error reading descriptions file: %v\n", descr)
		return
	}

	camera, err := gocv.VideoCaptureDevice(cameraID)
	if err != nil {
		fmt.Printf("Error opening camera: %v\n", cameraID)
		return
	}
	defer camera.Close()

	window := gocv.NewWindow("TensorFlow Open/Close")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	net := gocv.ReadNetFromTensorflow(model)
	if net.Empty() {
		fmt.Printf("Error reading model: %v\n", model)
		return
	}
	defer net.Close()

	status := "Ready"
	statusColor := color.RGBA{0, 100, 200, 0}
	fmt.Printf("Start reading camera: %v\n", cameraID)

	for {
		if ok := camera.Read(&img); !ok {
			fmt.Printf("Error reading camera: %d\n", cameraID)
			return
		}

		if img.Empty() {
			continue
		}

		croppedImage := img.Region(image.Rect(440, 0, 840, 720))
		blob := gocv.BlobFromImage(croppedImage, 1.0, image.Pt(64, 64), gocv.NewScalar(0, 0, 0, 0), true, false)

		net.SetInput(blob, "")
		prob := net.Forward("my_output/Softmax")
		probMat := prob.Reshape(1, 1)
		_, _, _, maxLoc := gocv.MinMaxLoc(probMat)

		desc := "Unknown"
		if maxLoc.X < 3 {
			desc = descriptions[maxLoc.X]
		}
		status = fmt.Sprintf("prediction: %v\n", desc)
		gocv.PutText(&img, status, image.Pt(450, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		blob.Close()
		prob.Close()
		probMat.Close()

		window.IMShow(croppedImage)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}