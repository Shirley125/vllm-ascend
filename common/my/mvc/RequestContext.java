package my.mvc;

import java.io.*;
import java.text.*;
import java.util.*;

import javax.servlet.*;
import javax.servlet.http.*;

import my.util.RequestUtils;
import my.util.ResourceUtils;

import org.apache.commons.beanutils.BeanUtils;
import org.apache.commons.beanutils.ConvertUtils;
import org.apache.commons.beanutils.Converter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.RandomStringUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

/**
 * 请求上下文
 * @author Winter Lau
 * @date 2010-1-13 下午04:18:00
 */
public class RequestContext {
	
	public final static String GLOBAL_USER_KEY = "g_user";
	
	private final static Log log = LogFactory.getLog(RequestContext.class);

	private final static int MAX_FILE_SIZE = 10*1024*1024; 
	private final static String UTF_8 = "UTF-8";
	
	private final static ThreadLocal<RequestContext> contexts = new ThreadLocal<RequestContext>();	
	private final static String upload_tmp_path;
	private final static String TEMP_UPLOAD_PATH_ATTR_NAME = "$NEXTAPP_TEMP_UPLOAD_PATH$";

	private static String webroot = null;
	
	private ServletContext context;
	private HttpSession session;
	private HttpServletRequest request;
	private HttpServletResponse response;
	private Map<String, Cookie> cookies;
	
	private final static Converter dt_converter = new Converter(){
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-M-d");
		SimpleDateFormat sdf_time = new SimpleDateFormat("yyyy-M-d H:m");
		@SuppressWarnings("rawtypes")
		public Object convert(Class type, Object value) {
			if(value == null) return null;
	        if (value.getClass().equals(type)) return value;
	        Date d = null;
	        try {
	            d = sdf_time.parse(value.toString());
	        } catch (ParseException e) {
	            try {
					d = sdf.parse(value.toString());
				} catch (ParseException e1) {
					return null;
				}
	        }
	        if(type.equals(java.util.Date.class))
	        	return d;
	        if(type.equals(java.sql.Date.class))
	        	return new java.sql.Date(d.getTime());
	        if(type.equals(java.sql.Timestamp.class))
	        	return new java.sql.Timestamp(d.getTime());
	        return null;
		}
	};
	
	static {
		webroot = getWebrootPath();
		//上传的临时目录
		upload_tmp_path = webroot + "WEB-INF" + File.separator + "tmp" + File.separator;
		try {
			FileUtils.forceMkdir(new File(upload_tmp_path));
		} catch (IOException excp) {}
		
		//BeanUtils对时间转换的初始化设置
		ConvertUtils.register(dt_converter, java.sql.Date.class);
		ConvertUtils.register(dt_converter, java.sql.Timestamp.class);
		ConvertUtils.register(dt_converter, java.util.Date.class);
	}
	
	private final static String getWebrootPath() {
		String root = RequestContext.class.getResource("/").getFile();
		try {
			if(root.endsWith(".svn/"))
				root = new File(root).getParentFile().getParentFile().getParentFile().getCanonicalPath();
			else
				root = new File(root).getParentFile().getParentFile().getCanonicalPath();
			root += File.separator;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return root;
	}
	
	/**
	 * 初始化请求上下文
	 * @param ctx
	 * @param req
	 * @param res
	 */
	public static RequestContext begin(ServletContext ctx, HttpServletRequest req, HttpServletResponse res) {
		RequestContext rc = new RequestContext();
		rc.context = ctx;
		rc.request = _AutoUploadRequest(_AutoEncodingRequest(req));
		rc.response = res;
		rc.response.setCharacterEncoding(UTF_8);
		rc.session = req.getSession(false);
		rc.cookies = new HashMap<String, Cookie>();
		Cookie[] cookies = req.getCookies();
		if(cookies != null)
			for(Cookie ck : cookies) {
				rc.cookies.put(ck.getName(), ck);
			}
		contexts.set(rc);
		return rc;
	}

	/**
	 * 返回Web应用的路径
	 * @return
	 */
	public static String root() { return webroot; }
	
	public static String getContextPath() {
		RequestContext ctx = RequestContext.get();
		return (ctx!=null)?ctx.contextPath():"";
	}
	
	/**
	 * 获取当前请求的上下文
	 * @return
	 */
	public static RequestContext get(){
		return contexts.get();
	}
	
	public void end() {
		String tmpPath = (String)request.getAttribute(TEMP_UPLOAD_PATH_ATTR_NAME);
		if(tmpPath != null){
			try {
				FileUtils.deleteDirectory(new File(tmpPath));
			} catch (IOException e) {
				log.fatal("Failed to cleanup upload directory: " + tmpPath, e);
			}
		}
		this.context = null;
		this.request = null;
		this.response = null;
		this.session = null;
		this.cookies = null;
		contexts.remove();
	}
	
	public Locale locale(){ return request.getLocale(); }

	/**
	 * 自动编码处理
	 * @param req
	 * @return
	 */
	private static HttpServletRequest _AutoEncodingRequest(HttpServletRequest req) {
		if(req instanceof RequestProxy)
			return req;
		HttpServletRequest auto_encoding_req = req;
		if("POST".equalsIgnoreCase(req.getMethod())){
			try {
				auto_encoding_req.setCharacterEncoding(UTF_8);
			} catch (UnsupportedEncodingException e) {}
		}
		else
			auto_encoding_req = new RequestProxy(req, UTF_8);
		
		return auto_encoding_req;
	}
	
	/**
	 * 自动文件上传请求的封装
	 * @param req
	 * @return
	 */
	private static HttpServletRequest _AutoUploadRequest(HttpServletRequest req){
		if(_IsMultipart(req)){
			String path = upload_tmp_path + RandomStringUtils.randomAlphanumeric(10);
			File dir = new File(path);
			if(!dir.exists() && !dir.isDirectory())	dir.mkdirs();
			try{
				req.setAttribute(TEMP_UPLOAD_PATH_ATTR_NAME,path);
				return new MultipartRequest(req, dir.getCanonicalPath(), MAX_FILE_SIZE, UTF_8);
			}catch(NullPointerException e){				
			}catch(IOException e){
				log.fatal("Failed to save upload files into temp directory: " + path, e);
			}
		}
		return req;
	}
	
	public int id() {
		return param("id", 0);
	}
	
	public String ip(){
		String ip = RequestUtils.getRemoteAddr(request);
		if(ip == null)
			ip = "127.0.0.1";
		return ip;
	}
	
	@SuppressWarnings("unchecked")
	public Enumeration<String> params() {
		return request.getParameterNames();
	}
	
	public String param(String name, String...def_value) {
		String v = request.getParameter(name);
		return (v!=null)?v:((def_value.length>0)?def_value[0]:null);
	}
	
	public long param(String name, long def_value) {
		return NumberUtils.toLong(param(name), def_value);
	}

	public int param(String name, int def_value) {
		return NumberUtils.toInt(param(name), def_value);
	}

	public byte param(String name, byte def_value) {
		return (byte)NumberUtils.toInt(param(name), def_value);
	}

	public String[] params(String name) {
		return request.getParameterValues(name);
	}

	public long[] lparams(String name){
		String[] values = params(name);
		if(values==null) return null;
		List<Long> lvs = new ArrayList<Long>();
		for(String v : values) {
			long lv = NumberUtils.toLong(v, Long.MIN_VALUE);
			if(lv != Long.MIN_VALUE && !lvs.contains(lvs))
				lvs.add(lv);
		}
		long [] llvs = new long[lvs.size()];
		for(int i=0;i<lvs.size();i++)
			llvs[i] = lvs.get(i);
		return llvs;
	}
	
	public String uri(){
		return request.getRequestURI();
	}
	
	public String contextPath(){
		return request.getContextPath();
	}
	
	public void redirect(String uri) throws IOException {
		response.sendRedirect(uri);
	}
	
	public void forward(String uri) throws ServletException, IOException {
		RequestDispatcher rd = context.getRequestDispatcher(uri);
		rd.forward(request, response);
	}

	public void include(String uri) throws ServletException, IOException {
		RequestDispatcher rd = context.getRequestDispatcher(uri);
		rd.include(request, response);
	}
	
	public boolean isUpload(){
		return (request instanceof MultipartRequest);
	}
	public File file(String fieldName) {
		if(request instanceof MultipartRequest)
			return ((MultipartRequest)request).getFile(fieldName);
		return null;
	}
	public boolean isRobot(){
		return RequestUtils.isRobot(request);
	}

	public ActionException fromResource(String bundle, String key, Object...args){
		String res = ResourceUtils.getStringForLocale(request.getLocale(), bundle, key, args);
		return new ActionException(res);
	}

	public ActionException error(String key, Object...args){		
		return fromResource("error", key, args);
	}
	
	/**
	 * 输出信息到浏览器
	 * @param msg
	 * @throws IOException
	 */
	public void printf(String fmt, Object...args) throws IOException {
		if(!UTF_8.equalsIgnoreCase(response.getCharacterEncoding()))
			response.setCharacterEncoding(UTF_8);
		response.getWriter().printf(fmt, args);
	}

	public void print(Object arg) throws IOException {
		if(!UTF_8.equalsIgnoreCase(response.getCharacterEncoding()))
			response.setCharacterEncoding(UTF_8);
		response.getWriter().print(arg);
	}
	
	public void output_json(String[] key, Object[] value) throws IOException {
		JSONObject jo = new JSONObject();
		for(int i=0;i<key.length;i++){
			if(value[i] instanceof Number)
				jo.put(key[i], (Number)value[i]);
			else if(value[i] instanceof Boolean)
				jo.put(key[i], (Boolean)value[i]);
			else
				jo.put(key[i], (String)value[i]);
		}
		print(JSON.toJSON(jo));
	}
	
	public static void main(String[] args){
		JSONObject jo = new JSONObject();
		for(int i=0;i<10;i++){
			jo.put("i"+i, i+"你\"好");
		}
		System.out.println(JSON.toJSON(jo));
	}

	public void output_json(String key, Object value) throws IOException {
		output_json(new String[]{key}, new Object[]{value});
	}

	public void json_msg(String msgkey, Object...args) throws IOException {
		output_json(
				new String[]{"error","msg"}, 
				new Object[]{0,ResourceUtils.getString("error", msgkey, args)}
		);
	}
	
	public void error(int code, String...msg) throws IOException {
		if(msg.length>0)
			response.sendError(code, msg[0]);
		else
			response.sendError(code);
	}
	
	public void forbidden() throws IOException { 
		error(HttpServletResponse.SC_FORBIDDEN); 
	}

	public void not_found() throws IOException { 
		error(HttpServletResponse.SC_NOT_FOUND); 
	}

	public ServletContext context() { return context; }
	public HttpSession session() { return session; }
	public HttpSession session(boolean create) { 
		return (session==null && create)?(session=request.getSession()):session; 
	}
	public Object sessionAttr(String attr) {
		HttpSession ssn = session();
		return (ssn!=null)?ssn.getAttribute(attr):null;
	}
	public HttpServletRequest request() { return request; }
	public HttpServletResponse response() { return response; }
	public Cookie cookie(String name) { return cookies.get(name); }
	public void cookie(String name, String value, int max_age, boolean all_sub_domain) {
		RequestUtils.setCookie(request, response, name, value, max_age, all_sub_domain);
	}
	public void deleteCookie(String name,boolean all_domain) { RequestUtils.deleteCookie(request, response, name, all_domain); }
	public String header(String name) { return request.getHeader(name); }
	public void header(String name, String value) { response.setHeader(name, value); }
	public void header(String name, int value) { response.setIntHeader(name, value); }
	public void header(String name, long value) { response.setDateHeader(name, value); }
	public String user_agent() { return header("user-agent"); }
	public int user_agent_code() {
		String ua = user_agent();
		return (ua!=null)?Math.abs(ua.hashCode()):0;
	}

	/**
	 * 设置public缓存，设置了此类型缓存要求此页面对任何人访问都是同样数据
	 * @param minutes 分钟
	 * @return
	 */
	public void setPublicCache(int minutes) {		
		if(!"POST".equalsIgnoreCase(request.getMethod())){
			int seconds = minutes * 60;
			header("Cache-Control","max-age="+seconds);
			Calendar cal = Calendar.getInstance(request.getLocale());
			cal.add(Calendar.MINUTE, minutes);
			header("Expires", cal.getTimeInMillis());
		}
	}
	
	/**
	 * 设置私有缓存
	 * @param minutes
	 * @return
	 */
	public void setPrivateCache(int minutes) {
		if(!"POST".equalsIgnoreCase(request.getMethod())){
			header("Cache-Control","private");
			Calendar cal = Calendar.getInstance(request.getLocale());
			cal.add(Calendar.MINUTE, minutes);
			header("Expires", cal.getTimeInMillis());
		}
	}

	/**
	 * 关闭缓存
	 */
	public void closeCache(){
        header("Pragma","must-revalidate, no-cache, private");
        header("Cache-Control","no-cache");
        header("Expires", "Sun, 1 Jan 2000 01:00:00 GMT");
	}
	
	/**
	 * 将HTTP请求参数映射到bean对象中
	 * @param req
	 * @param beanClass
	 * @return
	 * @throws Exception
	 */
	public <T> T form(Class<T> beanClass) {
		try{
			T bean = beanClass.newInstance();
			BeanUtils.populate(bean, request.getParameterMap());
			return bean;
		}catch(Exception e) {
			throw new ActionException(e.getMessage());
		}
	}
	
	/**
	 * 返回当前登录的用户资料
	 * @return
	 */
	public Object user() {
		return request.getAttribute(GLOBAL_USER_KEY);
	}
	
	/**
	 * 自动解码
	 * @author liudong
	 */
	private static class RequestProxy extends HttpServletRequestWrapper {
		private String uri_encoding; 
		RequestProxy(HttpServletRequest request, String encoding){
			super(request);
			this.uri_encoding = encoding;
		}
		
		/**
		 * 重载getParameter
		 */
		public String getParameter(String paramName) {
			String value = super.getParameter(paramName);
			return _DecodeParamValue(value);
		}

		/**
		 * 重载getParameterMap
		 */
		@SuppressWarnings({ "unchecked", "rawtypes" })
		public Map<String, Object> getParameterMap() {
			Map params = super.getParameterMap();
			HashMap<String, Object> new_params = new HashMap<String, Object>();
			Iterator<String> iter = params.keySet().iterator();
			while(iter.hasNext()){
				String key = (String)iter.next();
				Object oValue = params.get(key);
				if(oValue.getClass().isArray()){
					String[] values = (String[])params.get(key);
					String[] new_values = new String[values.length];
					for(int i=0;i<values.length;i++)
						new_values[i] = _DecodeParamValue(values[i]);
					
					new_params.put(key, new_values);
				}
				else{
					String value = (String)params.get(key);
					String new_value = _DecodeParamValue(value);
					if(new_value!=null)
						new_params.put(key,new_value);
				}
			}
			return new_params;
		}

		/**
		 * 重载getParameterValues
		 */
		public String[] getParameterValues(String arg0) {
			String[] values = super.getParameterValues(arg0);
			for(int i=0;values!=null&&i<values.length;i++)
				values[i] = _DecodeParamValue(values[i]);
			return values;
		}

		/**
		 * 参数转码
		 * @param value
		 * @return
		 */
		private String _DecodeParamValue(String value){
			if (StringUtils.isBlank(value) || StringUtils.isBlank(uri_encoding)
					|| StringUtils.isNumeric(value))
				return value;		
			try{
				return new String(value.getBytes("8859_1"), uri_encoding);
			}catch(Exception e){}
			return value;
		}

	}
	
	private static boolean _IsMultipart(HttpServletRequest req) {
		return ((req.getContentType() != null) && (req.getContentType()
				.toLowerCase().startsWith("multipart")));
	}

}
