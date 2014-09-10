package cn.nextapp.platform.beans;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;
import org.xmlpull.v1.XmlPullParserFactory;

/**
 * 插件接口URL定义
 * @author Liux
 */
public class URLs implements Serializable {

	private String catalogs;
	private String postList;
	private String postDetail;
	private String postPublish;
	private String postDelete;
	private String commentList;
	private String commentPublish;
	private String commentDelete;
	private String loginValidate;

	private URLs(){}
	public static URLs parse(InputStream in) throws IOException, XmlPullParserException  {
		URLs url = null;
		// 获得XmlPullParser解析器
		XmlPullParserFactory factory = XmlPullParserFactory.newInstance();
        factory.setNamespaceAware(true);
        XmlPullParser xmlParser = factory.newPullParser();
		try {
			xmlParser.setInput(in, "utf-8");
			// 获得解析到的事件类别，这里有开始文档，结束文档，开始标签，结束标签，文本等等事件。
			int evtType = xmlParser.getEventType();
			// 一直循环，直到文档结束
			while (evtType != XmlPullParser.END_DOCUMENT) {
				String tag = xmlParser.getName();
				switch (evtType) {

				case XmlPullParser.START_TAG:
					// 如果是标签开始，则说明需要实例化对象了
					if (tag.equalsIgnoreCase("urls")) {
						url = new URLs();
					}
					if(url != null){
			            if(tag.equalsIgnoreCase("catalog-list"))
			            	url.catalogs = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("post-list"))
			            	url.postList = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("post-detail"))
			            	url.postDetail = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("post-pub"))
			            	url.postPublish = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("post-delete"))
			            	url.postDelete = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("comment-list"))
			            	url.commentList = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("comment-pub"))
			            	url.commentPublish = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("comment-delete"))
			            	url.commentDelete = xmlParser.nextText();
			            else if(tag.equalsIgnoreCase("login-validate"))
			            	url.loginValidate = xmlParser.nextText();
					}
					break;
				case XmlPullParser.END_TAG:
					break;
				}
				// 如果xml没有结束，则导航到下一个节点
				evtType = xmlParser.next();
			}
		} finally {
			in.close();
		}
		return url;
	}
	
	public String getCatalogs() {
		return catalogs;
	}
	public String getPostList() {
		return postList;
	}
	public String getPostDetail() {
		return postDetail;
	}
	public String getPostPublish() {
		return postPublish;
	}
	public String getPostDelete() {
		return postDelete;
	}
	public String getCommentList() {
		return commentList;
	}
	public String getCommentPublish() {
		return commentPublish;
	}
	public String getCommentDelete() {
		return commentDelete;
	}
	public String getLoginValidate() {
		return loginValidate;
	}
	
	public void setCatalogs(String catalogs) { this.catalogs=catalogs; }
	public void setPostList(String postList) { this.postList=postList; }
	public void setPostDetail(String postDetail) { this.postDetail=postDetail; }
	public void setPostPublish(String postPublish) { this.postPublish=postPublish; }
	public void setPostDelete(String postDelete) { this.postDelete=postDelete; }
	public void setCommentList(String commentList) { this.commentList=commentList; }
	public void setCommentPublish(String commentPublish) { this.commentPublish=commentPublish; }
	public void setCommentDelete(String commentDelete) { this.commentDelete=commentDelete; }
	public void setLoginValidate(String loginValidate) { this.loginValidate=loginValidate; }
}
