from flask import Flask, request, render_template, Response
import client           #uncomment when integrating with the client.py API


if __name__ == '__main__':
    client.image_classify('/home/arya/Desktop/sample.jpg', 1)

