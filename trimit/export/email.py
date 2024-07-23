import sendgrid
from sendgrid.helpers.mail import (
    Mail,
    Email,
    To,
    Content,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)
import base64
import os
from trimit.models.backend_models import ExportResults
import mimetypes


def parse_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return mime_type


def in_ignore_extensions(file_path: str) -> bool:
    ignore_extensions = [".p"]
    return any(file_path.endswith(ext) for ext in ignore_extensions)


def send_email_with_export_results(
    workflow_id: str, user_email: str, export_results: ExportResults
):
    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    workflow_id = workflow_id
    video_builder_page_link = f"https://app.trimit.ai/builder?projectId={workflow_id}"
    message = Mail(
        from_email=os.environ["TRIMIT_FROM_EMAIL_ADDRESS"],
        to_emails=user_email,
        subject="Your Trimit video edits are Ready!",
        html_content=f'<strong>TrimIt created the video edits in the attached files for you. You can also stream/edit the video <a href="{video_builder_page_link}">here</a></strong>',
    )

    for field in export_results.model_fields:
        filepath = getattr(export_results, field)
        # TODO handle lists/dicts of filepaths
        if (
            filepath is None
            or not isinstance(filepath, str)
            or in_ignore_extensions(filepath)
        ):
            continue
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            continue
        with open(filepath, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
        attachment = Attachment()
        attachment.file_content = FileContent(encoded)
        attachment.file_type = FileType(parse_mime_type(filepath))
        attachment.file_name = FileName(os.path.basename(filepath))
        attachment.disposition = Disposition("attachment")
        message.add_attachment(attachment)

    try:
        response = sg.send(message)
    except Exception as e:
        print(e)
    else:
        return response
