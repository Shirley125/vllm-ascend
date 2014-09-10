package my.view;

import java.io.*;
import java.util.*;

import javax.servlet.*;
import javax.servlet.http.*;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * 用于URL地址的转换
 * http://www.youcity.cn/news/list/1/3 -> {base}/news/list.vm?p1=1&p2=3
 * @author liudong
 */
public final class URLMappingServlet extends HttpServlet {

	private final static Log log = LogFactory.getLog(URLMappingServlet.class);	
	
	public final static String CURRENT_URI = "current_uri";	//{index}
	public final static String REQUEST_URI = "request_uri";	//{/index}
	
	private final static String DEFAULT_INDEX_PAGE = "index.vm";
	private final static String PAGE_EXTENSION = ".vm";
	private final static char URL_SEPERATOR = '/';

	private String default_base;
	private HashMap<String, String> other_base = new HashMap<String, String>();
	
	private String rootDomain = "oschina.net";
	
	@Override
	@SuppressWarnings("unchecked")
	public void init() throws ServletException {
		Enumeration<String> names = getInitParameterNames();
		while(names.hasMoreElements()){
			String name = names.nextElement();
			String v = getInitParameter(name);
			if("default".equalsIgnoreCase(name)){
				default_base = v;
				continue;
			}
			for(String n : StringUtils.split(name, ',')){
				other_base.put(n, v);
			}
		}
	}

	private String _GetTemplateBase(HttpServletRequest req) {
		String base = null;
		String prefix = req.getServerName().toLowerCase();
		base = other_base.get(prefix);
		if(base != null)
			return base;
		int idx = prefix.indexOf(rootDomain);
		if(idx > 0){
			prefix = prefix.substring(0, idx - 1);
			base = other_base.get(prefix);
		}
		return (base==null)?default_base:base;
	}
	
	/**
	 * 执行页面映射过程
	 * @param req
	 * @param res
	 * @throws ServletException
	 * @throws IOException
	 */
	protected void perform(HttpServletRequest req, HttpServletResponse res)
			throws ServletException, IOException {

		StringBuilder show_page = new StringBuilder(_GetTemplateBase(req));
		String prefix = req.getServletPath().substring(1);
		String spath = req.getRequestURI().substring(req.getContextPath().length());	
		req.setAttribute(REQUEST_URI, spath);
		req.setAttribute(CURRENT_URI, prefix);
		//解析URL地址
		String[] s_result = spath.substring(1).split(String.valueOf(URL_SEPERATOR));
		if(s_result.length==1){
			show_page.append(prefix);
			show_page.append(URL_SEPERATOR);
			show_page.append(DEFAULT_INDEX_PAGE);
		}
		else{
			show_page.append(prefix);
			show_page.append(URL_SEPERATOR);
			/* Ex: http://www.71way.com/admin/login/ld */
			StringBuilder testPath = new StringBuilder(show_page);
			testPath.append(s_result[1]);
			testPath.append(PAGE_EXTENSION);
			boolean isVM = _IsVmExist(testPath.toString());
			int param_idx = 1;
			if(isVM){
				show_page.append(s_result[1]);
				show_page.append(PAGE_EXTENSION);
				param_idx = 2;
			}
			else{
				show_page.append(DEFAULT_INDEX_PAGE);
			}
			for(int i=param_idx;i<s_result.length;i++){
				if(i==param_idx) 
					show_page.append('?');
				else
					show_page.append('&');
				show_page.append('p');
				show_page.append((i-param_idx+1));
				show_page.append('=');
				show_page.append(s_result[i]);
			}
			testPath.setLength(0);
			testPath = null;
		}
		if(log.isDebugEnabled())
			log.debug("request_uri="+spath+",servlet_path="+req.getServletPath()+",vm="+show_page);
		//执行真实的页面
		RequestDispatcher rd = getServletContext().getRequestDispatcher(show_page.toString());
		rd.forward(req, res);	
		
	}

	private final static List<String> vm_cache = new ArrayList<String>();
	
	/**
	 * 判断某个页面是否存在，如果存在则缓存此结果
	 * @param path
	 * @return
	 */
	private boolean _IsVmExist(String path){
		if(vm_cache.contains(path))
			return true;
		File testFile = new File(getServletContext().getRealPath(path));
		boolean isVM = testFile.exists() && testFile.isFile();
		if(isVM){
			synchronized(vm_cache){
				if(!vm_cache.contains(path))
					vm_cache.add(path);
			}
		}
		return isVM;
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp)
			throws ServletException, IOException {
		perform(req, resp);
	}

	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp)
			throws ServletException, IOException {
		perform(req, resp);
	}

}
